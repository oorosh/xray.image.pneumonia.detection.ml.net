using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace pneumonia.detection
{
    class Program
    {
        static void Main()
        {
            //Dataset taken from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/
            const string datasetUrl = "https://storage.googleapis.com/kaggle-data-sets/17810/23812/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587921765&Signature=HxSUJTlJVYahLDVTXsjq0q15M0RdUlhVeAAAcHHDgfO15r9T9vNrDbVQdac%2BZ7vn5FAJJLxRaOlWUVRQIcf5P2fwF0D6vM3mMdB76x6aukiFu6kSRYL8XWbM2Kw8aPjazhmvSnVJpRTFAlycpfh7kqU2g%2F8kEEgo9De4lU7hDZ85RtjQfajPxpLmgdrOlxTbGmm8fQaZ4DJgqDpU00cLKQAJSTFsBraM3xZcZMFu4tf6GKyUXSq%2FFR08h%2BOkQuUOL5xFI2aLweFSgKGMxfDawjXqNK5O3yqaLtDVapEMbT%2BzopDQeC0XabTysK6J4aktdrMhnSLh0E2MKSpzpCBBmg%3D%3D&response-content-disposition=attachment%3B+filename%3Dchest-xray-pneumonia.zip";
            const string datasetFullName = "chest-xray-pneumonia.zip";

            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../"));

            var datasetFolder = Path.Combine(projectDirectory, "dataset");

            // 1. Download the image set and unzip
            Utility.DownloadImageSet(datasetFolder, datasetUrl, datasetFullName);

            // Measuring training time
            var watch = Stopwatch.StartNew();

            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");

            datasetFolder = Path.Combine(projectDirectory, "chest_xray");
            var trainImageFolderPath = Path.Combine(datasetFolder, "train");
            var testImageFolderPath = Path.Combine(datasetFolder, "test");
            var validationImageFolderPath = Path.Combine(datasetFolder, "val");

            var context = new MLContext(seed: 1);

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            var trainImageDataShuffle = Utility.GetTrainImageData(trainImageFolderPath, context);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            trainImageDataShuffle = Utility.TransformDataView(context, trainImageDataShuffle, trainImageFolderPath);

            // Load test images from files to memory
            var testImages = Utility.LoadImagesFromDirectory(testImageFolderPath, true);
            var testImageDataView = context.Data.LoadFromEnumerable(testImages);
            testImageDataView = Utility.TransformDataView(context, testImageDataView, testImageFolderPath);

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250
                // here instead of ResnetV2101 you can try a different 
                // architecture/ pre-trained model. 
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                //Epoch = 50,
                //BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testImageDataView,
                // Disable EarlyStopping to run to specified number of epochs.
                //EarlyStoppingCriteria = null,
                ReuseTrainSetBottleneckCachedValues = true,
                WorkspacePath = workspaceRelativePath
            };


            //4. Define the model's training pipeline using DNN default values
            var pipeline = context.MulticlassClassification.Trainers
               .ImageClassification(classifierOptions)
               .Append(context.Transforms.Conversion.MapKeyToValue(
                   outputColumnName: "PredictedLabel",
                   inputColumnName: "PredictedLabel"));

            Console.WriteLine("*** Training the image classification model " +
              "with DNN Transfer Learning on top of the selected " +
              "pre-trained model/architecture ***");

            //5. Train
            var model = pipeline.Fit(trainImageDataShuffle);

            Console.WriteLine("Training with transfer learning finished.");           
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {(elapsedMs / 1000) / 60} minutes.");

            //6. Evaluate the trained model on the passed test dataset.
            Console.WriteLine("Evaluate model with validation data.\n");
            var valImages = Utility.LoadImagesFromDirectory(validationImageFolderPath, true);
            var valImageDataView = context.Data.LoadFromEnumerable(valImages);

            valImageDataView = Utility.TransformDataView(context, valImageDataView, validationImageFolderPath);

            var metric = context.MulticlassClassification.Evaluate(model.Transform(valImageDataView));

            Console.WriteLine($"LogLoss : " + metric.LogLoss + "\n" +
                $"MacroAccuracy : " + metric.MacroAccuracy + "\n" +
                $"MicroAccuracy : " + metric.MicroAccuracy + "\n");

            context.Model.Save(model, trainImageDataShuffle.Schema, "model.zip");

            Console.ReadLine();
        }
    }
}
