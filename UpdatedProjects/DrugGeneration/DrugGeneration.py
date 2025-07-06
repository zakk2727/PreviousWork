
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ast
import logging
import gc

import io

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Create 4 virtual GPUs with 11GB memory each
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=11264),
                                                             tf.config.LogicalDeviceConfiguration(memory_limit=11264),
                                                             tf.config.LogicalDeviceConfiguration(memory_limit=11264),
                                                             tf.config.LogicalDeviceConfiguration(memory_limit=11264)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
#print(tf.config.list_logical_devices('GPU'))

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy(logical_gpus)

"""

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

"""

df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv")
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
#print(df.head(5))

print("Size of dataframe: ", len(df))

'''
connection_url = URL.create(drivername="mssql+pyodbc",
                            username="BotTrainingUser",
                            password="Banana27!",
                            host="172.31.64.1",
                            port=1433,
                            database="master",
                            query={"driver":"ODBC Driver 18 for SQL Server",
                                   "TrustServerCertificate":"yes",
                                   })

engine = create_engine(connection_url)

conn = engine.connect()

conn.close()
'''

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'

bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

MAX_MOLSIZE = max(df["smiles"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 15

VAE_LR = 5e-4
NUM_ATOMS = int(120)  # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = int(4 + 1)  # Number of bond types
LATENT_DIM = int(435)  # Size of the latent space

def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS))
    features = np.zeros((NUM_ATOMS, ATOM_DIM))

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where((np.argmax(features, axis=1) != ATOM_DIM - 1) & (np.sum(adjacency[:-1], axis=(0, 1)) != 0))[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule

TrainDF, TestDF = train_test_split(df, test_size=0.20, random_state=27)
TrainDF, ValidationDF = train_test_split(TrainDF, test_size=0.10, random_state=27)

TrainDF.reset_index(drop=True, inplace=True)
TestDF.reset_index(drop=True, inplace=True)
ValidationDF.reset_index(drop=True, inplace=True)

print("Size of TrainDF: ", len(TrainDF))
print("Size of TestDF: ", len(TestDF))
print("Size of ValidationDF: ", len(ValidationDF))

#train_df = df.sample(frac=0.75, random_state=27)  # random state is a seed value
#train_df.reset_index(drop=True, inplace=True)

train_adjacency_tensor, train_feature_tensor, train_qed_tensor = [], [], []
for idx in range(10240):
    adjacency, features = smiles_to_graph(TrainDF.loc[idx]["smiles"])
    qed = TrainDF.loc[idx]["qed"]
    train_adjacency_tensor.append(adjacency)
    train_feature_tensor.append(features)
    train_qed_tensor.append(qed)
    
del TrainDF
gc.collect()
    
test_adjacency_tensor, test_feature_tensor, test_qed_tensor = [], [], []
for idx in range(2560):
    adjacency, features = smiles_to_graph(TestDF.loc[idx]["smiles"])
    qed = TestDF.loc[idx]["qed"]
    test_adjacency_tensor.append(adjacency)
    test_feature_tensor.append(features)
    test_qed_tensor.append(qed)
    
del TestDF
gc.collect()
    
Validation_adjacency_tensor, Validation_feature_tensor, Validation_qed_tensor = [], [], []
for idx in range(1024):
    adjacency, features = smiles_to_graph(ValidationDF.loc[idx]["smiles"])
    qed = ValidationDF.loc[idx]["qed"]
    Validation_adjacency_tensor.append(adjacency)
    Validation_feature_tensor.append(features)
    Validation_qed_tensor.append(qed)
    
del ValidationDF
gc.collect()

train_adjacency_tensor = np.array(train_adjacency_tensor,dtype='float32')
train_feature_tensor = np.array(train_feature_tensor,dtype='float32')
train_qed_tensor = np.array(train_qed_tensor,dtype='float32')

test_adjacency_tensor = np.array(test_adjacency_tensor,dtype='float32')
test_feature_tensor = np.array(test_feature_tensor,dtype='float32')
test_qed_tensor = np.array(test_qed_tensor,dtype='float32')

Validation_adjacency_tensor = np.array(Validation_adjacency_tensor,dtype='float32')
Validation_feature_tensor = np.array(Validation_feature_tensor,dtype='float32')
Validation_qed_tensor = np.array(Validation_qed_tensor,dtype='float32')

print("Training adjacency shape: ", train_adjacency_tensor.shape)
print("Training feature shape: ", train_feature_tensor.shape)
print("Training qed shape: ", train_qed_tensor.shape)

print("Test adjacency shape: ", test_adjacency_tensor.shape)
print("Test feature shape: ", test_feature_tensor.shape)
print("Test qed shape: ", test_qed_tensor.shape)

print("Validation adjacency shape: ", Validation_adjacency_tensor.shape)
print("Validation feature shape: ", Validation_feature_tensor.shape)
print("Validation qed shape: ", Validation_qed_tensor.shape)
#print(qed_tensor[0])

TrainAdjacencyDS = tf.data.Dataset.from_tensor_slices(train_adjacency_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
TrainFeatureDS = tf.data.Dataset.from_tensor_slices(train_feature_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
TrainQedDS = tf.data.Dataset.from_tensor_slices(train_qed_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)

#print(tf.data.DatasetSpec.from_value(TrainAdjacencyDS))
#print(tf.data.DatasetSpec.from_value(TrainFeatureDS))
#print(tf.data.DatasetSpec.from_value(TrainQedDS))

del train_adjacency_tensor, train_feature_tensor, train_qed_tensor
gc.collect()

TrainDS = tf.data.Dataset.zip((TrainAdjacencyDS, TrainFeatureDS, TrainQedDS)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

del TrainAdjacencyDS, TrainFeatureDS, TrainQedDS
gc.collect()

TestAdjacencyDS = tf.data.Dataset.from_tensor_slices(test_adjacency_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
TestFeatureDS = tf.data.Dataset.from_tensor_slices(test_feature_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
TestQedDS = tf.data.Dataset.from_tensor_slices(test_qed_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)

del test_adjacency_tensor, test_feature_tensor, test_qed_tensor
gc.collect()

TestDS = tf.data.Dataset.zip((TestAdjacencyDS, TestFeatureDS, TestQedDS)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

del TestAdjacencyDS, TestFeatureDS, TestQedDS
gc.collect()

ValidationAdjacencyDS = tf.data.Dataset.from_tensor_slices(Validation_adjacency_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
ValidationFeatureDS = tf.data.Dataset.from_tensor_slices(Validation_feature_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)
ValidationQedDS = tf.data.Dataset.from_tensor_slices(Validation_qed_tensor).batch(GLOBAL_BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE)

del Validation_adjacency_tensor, Validation_feature_tensor, Validation_qed_tensor
gc.collect()

ValidationDS = tf.data.Dataset.zip((ValidationAdjacencyDS, ValidationFeatureDS, ValidationQedDS)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

del ValidationAdjacencyDS, ValidationFeatureDS, ValidationQedDS
gc.collect()

#TrainDS = tf.data.Dataset.from_tensor_slices([adjacency_tensor, feature_tensor, qed_tensor]).batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

for (example_adjacency, example_feature, example_qed) in TrainDS.take(1):
    #print('Example Adjacency: ', example_adjacency.numpy()[:1])
    print('Example Adjacency Shape: ', example_adjacency.shape)
    print()
    #print('Example Feature: ', example_feature.numpy()[:1])
    print('Example Feature Shape: ', example_feature.shape)
    print()
    #print('Example QED: ', example_qed.numpy()[:1])
    print('Example QED Shape: ', example_qed.shape)

#TestGraphLayer = RelationalGraphConvLayer()
#TestGraphLayerResult = TestGraphLayer([example_adjacency,example_feature])

#print(TestGraphLayer([example_adjacency,example_feature]))

'''
def get_encoder(gconv_units, latent_dim, adjacency_shape, feature_shape, dense_units, dropout_rate):
    adjacency = tf.keras.layers.Input(shape=adjacency_shape)
    features = tf.keras.layers.Input(shape=feature_shape)

    # Propagate through one or more graph convolutional layers
    features_transformed = features
    for units in gconv_units:
        features_transformed = RelationalGraphConvLayer(units)([adjacency, features_transformed])
    # Reduce 2-D representation of molecule to 1-D
    x = tf.keras.layers.GlobalAveragePooling1D()(features_transformed)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    z_mean = tf.keras.layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
    log_var = tf.keras.layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

    encoder = tf.keras.Model([adjacency, features], [z_mean, log_var], name="encoder")

    return encoder


def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="tanh")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
    #print(adjacency_shape)
    #print(tf.math.reduce_prod(adjacency_shape))
    #print(tf.math.reduce_prod(adjacency_shape).numpy())

    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_adjacency = tf.keras.layers.Dense(tf.math.reduce_prod(adjacency_shape).numpy())(x)
    x_adjacency = tf.keras.layers.Reshape(adjacency_shape)(x_adjacency)
    print(x_adjacency)
    print(tf.convert_to_tensor(x_adjacency, dtype=tf.float32))
    print(x_adjacency.shape)
    
    # Symmetrify tensors in the last two dimensions
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
    x_adjacency = tf.keras.layers.Softmax(axis=1)(x_adjacency)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_features = tf.keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_features = tf.keras.layers.Reshape(feature_shape)(x_features)
    x_features = tf.keras.layers.Softmax(axis=2)(x_features)

    decoder = tf.keras.Model(latent_inputs, outputs=[x_adjacency, x_features], name="decoder")

    return decoder
'''

from CustomModelClasses.MoleculeGenerator import MoleculeGenerator
from CustomModelClasses.MoleculeVAEEncoder import MoleculeGeneratorEncoder
from CustomModelClasses.MoleculeVAEDecoder import MoleculeGeneratorDecoder
from CustomModelClasses.Sampling import Sampling
from CustomModelClasses.RelationalGraphConvLayer import RelationalGraphConvLayer


#vae_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(vae_optimizer)



'''
encoder = get_encoder(
    gconv_units=[9],
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
    latent_dim=LATENT_DIM,
    dense_units=[512],
    dropout_rate=0.0,
)
decoder = get_decoder(
    dense_units=[128, 256, 512],
    dropout_rate=0.2,
    latent_dim=LATENT_DIM,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)

enc_output_mean, enc_output_log_var = encoder([example_adjacency, example_feature])
#print(enc_output_mean)
#print(enc_output_log_var)

sampler_out = sampler([enc_output_mean,enc_output_log_var])
#print(sampler_out)

dec_output = decoder(sampler_out)
#print(dec_output)

'''

# Distribute the datasets
train_dist_dataset = strategy.experimental_distribute_dataset(TrainDS)
test_dist_dataset = strategy.experimental_distribute_dataset(TestDS)
validation_dist_dataset = strategy.experimental_distribute_dataset(ValidationDS)

with strategy.scope():
    sampling_layer = Sampling()
    gconv_layers = [RelationalGraphConvLayer() for _ in range(8)]
    encoder = MoleculeGeneratorEncoder(gconv_units=[8],gconv_layers=gconv_layers,latent_dim=LATENT_DIM,adjacency_shape=[BOND_DIM, NUM_ATOMS, NUM_ATOMS],feature_shape=[NUM_ATOMS, ATOM_DIM],dense_units=[512],dropout_rate=0.0)
    decoder = MoleculeGeneratorDecoder(dense_units=[128, 256, 512],dropout_rate=0.2,latent_dim=LATENT_DIM,adjacency_shape=[BOND_DIM, NUM_ATOMS, NUM_ATOMS],feature_shape=[NUM_ATOMS, ATOM_DIM])
    model = MoleculeGenerator(encoder, decoder, sampling_layer, MAX_MOLSIZE)
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=VAE_LR)
    model.compile(optimizer=vae_optimizer)
    #built_data = model([example_adjacency, example_feature])
    model.build()
    

print(model.built)

print(model.summary())
history = model.fit(train_dist_dataset, epochs=EPOCHS,validation_data=validation_dist_dataset)

def inference(model, batch_size):
    z = tf.keras.random.normal((batch_size, LATENT_DIM))
    reconstruction_adjacency, reconstruction_features = model.predict(z)
    # obtain one-hot encoded adjacency tensor
    adjacency = tf.keras.ops.argmax(reconstruction_adjacency, axis=1)
    adjacency = tf.keras.ops.one_hot(adjacency, depth=BOND_DIM, axis=1)
    # Remove potential self-loops from adjacency
    #adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    adjacency = adjacency * (1.0 - tf.keras.ops.eye(NUM_ATOMS, dtype="float32")[None, None])
    # obtain one-hot encoded feature tensor
    features = tf.keras.ops.argmax(reconstruction_features, axis=2)
    features = tf.keras.ops.one_hot(features, depth=ATOM_DIM, axis=2)
    return [graph_to_molecule([adjacency[i].numpy(), features[i].numpy()]) for i in range(batch_size)]

molecules = inference(model, 1000)

MolsToGridImage([m for m in molecules if m is not None][:1000], molsPerRow=5, subImgSize=(260, 160))

def plot_latent(vae, data, labels):
    # display a 2D plot of the property in the latent space
    z_mean, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


#plot_latent(model, [adjacency_tensor[:8000], feature_tensor[:8000]], qed_tensor[:8000])