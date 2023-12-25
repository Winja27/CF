using Microsoft.ML;
using Microsoft.ML.Trainers;
using Newtonsoft.Json.Linq;


namespace MovieRecommendation
{

    class Program
    {
        static void Main(string[] args)
        {
            int epoch = 20;  // 迭代次数
            int retainedFactors = 100;    // 保留的因子数量
            int fold = 10;  // k-fold折数

            MLContext mlContext = new MLContext();

            // 加载数据并k-fold
            var folds = LoadData(mlContext, fold);
            double[] maes = new double[folds.Count];
            double[] rmses = new double[folds.Count];
            double[] mses = new double[folds.Count];
            for (int i = 0; i < folds.Count; i++)
            {
                // 构建并训练模型
                ITransformer model = BuildAndTrainModel(mlContext, folds[i].TrainSet, epoch, retainedFactors);

                // 评估模型
                (double mae, double rmse, double mse) = EvaluateModel(mlContext, folds[i].TestSet, model);

                // Store the evaluation metrics
                maes[i] = mae;
                rmses[i] = rmse;
                mses[i] = mse;
            }

            // 评估结果
            Console.WriteLine("=============== Metrics after K-Fold ===============");
            Console.WriteLine($"Mean Absolute Error (MAE): {maes.Average()}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {rmses.Average()}");
            Console.WriteLine($"Mean Squared Error (MSE): {mses.Average()}");

            // 写入 CSV 文件
            using (StreamWriter writer = new StreamWriter("../../../metrics.csv"))
            {
                writer.WriteLine("MAE, RMSE,MSE");
                writer.WriteLine($"{maes.Average()},{rmses.Average()},{mses.Average()}");
            }
        }

    // 加载数据
    public static IReadOnlyList<DataOperationsCatalog.TrainTestData> LoadData(MLContext mlContext, int fold)
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "ratings.csv");
            IDataView dataView = mlContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');
            // k-fold
            var folds = mlContext.Data.CrossValidationSplit(dataView, numberOfFolds: fold);
            return folds;
        }

        // 构建并训练模型
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView, int epoch, int retainedFactors)
        {
            // Add data transformations
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            // Set algorithm options and append algorithm
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = epoch,
                ApproximationRank = retainedFactors
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        // 评估模型
        public static (double, double, double) EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("MAE: " + metrics.MeanAbsoluteError.ToString());
            Console.WriteLine("RMSE: " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("MSE: " + metrics.MeanSquaredError.ToString());
            Console.WriteLine();
            return (metrics.MeanAbsoluteError, metrics.RootMeanSquaredError, metrics.MeanSquaredError);
        }
    }
}
