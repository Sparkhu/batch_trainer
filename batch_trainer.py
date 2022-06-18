from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.ml.feature import MinMaxScaler, MinMaxScalerModel
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import array
from pyspark.sql.functions import percent_rank
from pyspark.sql import Window
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch

class Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias = True)


    def reset_hidden_state(self):
        self.hidden = (
                torch.randn(self.layers, self.seq_len, self.hidden_dim, requires_grad=True),
                torch.randn(self.layers, self.seq_len, self.hidden_dim, requires_grad=True))


    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
sc = init_orca_context(
    cluster_mode="local", cores=4, num_nodes=3, memory="4g",
    driver_memory="4g", driver_cores=1, worker_memory="4g",
)
print("################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################")
# hyperparameter
TIMESTEP = 60
N_OUT = 20
N_BATCH = 64
N_EPOCH=100
N_HIDDEN = 60
N_LAYER = 1
learning_rate = 0.01
train_ratio = 0.8

# Parse Args
import argparse
import datetime

parser = argparse.ArgumentParser()

# Target Feature
parser.add_argument("db")
parser.add_argument("table")
parser.add_argument("feature")

# Train Interval
parser.add_argument("start", type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d_%H:%M:%S')) # ex) 2012-01-01_12:00:00
parser.add_argument("end", type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d_%H:%M:%S'))

parser.add_argument("--timestep", type=int)
parser.add_argument("--num_out", type=int)

args = parser.parse_args()

DB = args.db
TABLE = args.table
FEATURE = args.feature
START = args.start
END = args.end
if args.timestep:
    TIMESTEP = args.timestep
if args.num_out:
    N_OUT = args.num_out

# Spark session
spark = OrcaContext.get_spark_session()

# Read batch data
current = START
dfs = []
while current <= END:
    # Read table data in the 'current' day
    try:
        path = f'hdfs:///data/{DB}_{TABLE}/year={current.year}/month={current.strftime("%m")}/day={current.strftime("%d")}'
        print("file read: ", path)
        df = spark.read.format('org.apache.spark.sql.json').load(path)
        dfs.append(df)
    except:
        pass
    current += datetime.timedelta(days=1)

df_ = reduce(DataFrame.unionAll, dfs)


# Normalize data
df = df_.select('time', FEATURE)
assembler = VectorAssembler(inputCols=[FEATURE],outputCol=FEATURE + "_vec")
df = assembler.transform(df)
mmScaler = MinMaxScaler(outputCol=FEATURE + "_scaled")
mmScaler.setInputCol(FEATURE + "_vec")
mmModel = mmScaler.fit(df)
df = mmModel.transform(df)
scalarize_udf = F.udf(lambda x: float(x[0]), FloatType())
df = df.withColumn(FEATURE + "_scaled", array(scalarize_udf(FEATURE + "_scaled")))
# Save scaler model
minMaxScalerPath = f"hdfs:///modelDir/{DB}_{TABLE}_{FEATURE}/minmaxScaler"
mmScaler.write().overwrite().save(minMaxScalerPath)
minMaxModelPath = f"hdfs:///modelDir/{DB}_{TABLE}_{FEATURE}/minmaxModel"
mmModel.write().overwrite().save(minMaxModelPath)

# Split train and test data
df = df.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("time")))
train_df = df.where("rank <= " +str(train_ratio)).select("time", FEATURE + "_scaled")
test_df = df.where("rank > " +str(train_ratio)).select("time", FEATURE + "_scaled")


# Sliding window
sliding_window = Window.orderBy(F.col("time").desc()).rowsBetween(Window.currentRow, TIMESTEP + N_OUT - 1)
x_udf = F.udf(lambda l: l[:TIMESTEP] ,ArrayType(ArrayType(FloatType())))
y_udf = F.udf(lambda l: l[TIMESTEP:] ,ArrayType(ArrayType(FloatType())))

train_df_id = train_df.withColumn(FEATURE + "_collect", F.collect_list(FEATURE + "_scaled").over(sliding_window)) \
    .withColumn(FEATURE + "_collect", F.collect_list(FEATURE + "_scaled").over(sliding_window)) \
    .withColumn("index", monotonically_increasing_id())
len = train_df_id.select(F.max("index")).first()[0]
train_df_cut = train_df_id.where(F.col("index") <= len - TIMESTEP - N_OUT) \
    .select(x_udf(FEATURE + "_collect").alias("X"), y_udf(FEATURE + "_collect").alias("Y"))

test_df_id = test_df.withColumn(FEATURE + "_collect", F.collect_list(FEATURE + "_scaled").over(sliding_window)) \
    .withColumn(FEATURE + "_collect", F.collect_list(FEATURE + "_scaled").over(sliding_window)) \
    .withColumn("index", monotonically_increasing_id())
len = test_df_id.select(F.max("index")).first()[0]
test_df_cut = test_df_id.where(F.col("index") <= len - TIMESTEP - N_OUT) \
    .select(x_udf(FEATURE + "_collect").alias("X"), y_udf(FEATURE + "_collect").alias("Y"))


# build dataset
def build_dataset(df):
    return np.array(df.select("X").rdd.map(lambda r: r[0]).collect()), np.array(df.select("Y").rdd.map(lambda r: r[0]).collect())

trainX, trainY = build_dataset(train_df_cut)
testX, testY = build_dataset(test_df_cut)

# Make tensor
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

dataset = TensorDataset(trainX_tensor, trainY_tensor)
dataloader = DataLoader(dataset,
                        batch_size=N_BATCH,
                        shuffle=True,
                        drop_last=True)
testset = TensorDataset(testX_tensor, testY_tensor)
testloader = DataLoader(testset,
                        batch_size=N_BATCH,
                        shuffle=False)

# Instantiate model
model = Net(1, N_HIDDEN, TIMESTEP, N_OUT, N_LAYER)
model.train()
criterion  = nn.MSELoss()
adam = torch.optim.Adam(model.parameters(), learning_rate)

# Bind model into Bigdl Orca estimator
model_dir = f'/modelDir/offline/{DB}_{TABLE}_{FEATURE}/'
est = Estimator.from_torch(model=model, optimizer=adam, loss=criterion, model_dir=model_dir)

# Train model
est.fit(data=dataloader, epochs=N_EPOCH,  checkpoint_trigger=EveryEpoch())

# Sava Model in HDFS
model_obj = est.get_model()
torch.save(model_obj.state_dict(), f'/tmp/offlineModel_{DB}_{TABLE}_{FEATURE}.pt') # Memory -> Local
# Todo: Now subprocess call is envrironment dependent. SO it has to be substituted.
import subprocess # Local -> HDFS :
subprocess.call(["hdfs", "dfs", "-copyFromLocal", "-f", f"/tmp/offlineModel_{DB}_{TABLE}_{FEATURE}.pt", f"/modelDir/offline/{DB}_{TABLE}_{FEATURE}.pt"])

stop_orca_context()
