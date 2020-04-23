using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using pneumonia.detection.common;
using System;
using System.Collections.Generic;
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

            //Download data set and unzip it
            DownloadImageSet(datasetFolder, datasetUrl, datasetFullName);

            // Measuring training time
            var watch = Stopwatch.StartNew();

            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var trainImageFolderPath = Path.Combine(datasetFolder, "chest_xray", "train");
            var testImageFolderPath = Path.Combine(datasetFolder, "chest_xray", "test");
            var validationImageFolderPath = Path.Combine(datasetFolder, "chest_xray", "val");
            
            var trainImages = LoadImagesFromDirectory(trainImageFolderPath, true);

            var context = new MLContext(seed: 1);

            var trainImageData = context.Data.LoadFromEnumerable(trainImages);
            var trainImageDataShuffle = context.Data.ShuffleRows(trainImageData);

            // Apply transforms to the input dataset:
            // MapValueToKey : map 'string' type labels to keys
            // LoadImages : load raw images to "Image" column
            trainImageDataShuffle = TransformDataView(context, trainImageDataShuffle, trainImageFolderPath);

            var testImages = LoadImagesFromDirectory(testImageFolderPath, true);
            var testImageDataView = context.Data.LoadFromEnumerable(testImages);

            testImageDataView = TransformDataView(context, testImageDataView, testImageFolderPath);            

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
                ReuseTrainSetBottleneckCachedValues = true,
                EarlyStoppingCriteria = null,
                WorkspacePath = workspaceRelativePath                
            };

            var pipeline = context.MulticlassClassification.Trainers
               .ImageClassification(classifierOptions)
               .Append(context.Transforms.Conversion.MapKeyToValue(
                   outputColumnName: "PredictedLabel",
                   inputColumnName: "PredictedLabel"));

            Console.WriteLine("*** Training the image classification model " +
              "with DNN Transfer Learning on top of the selected " +
              "pre-trained model/architecture ***");

            var model = pipeline.Fit(trainImageDataShuffle);

            Console.WriteLine("Training with transfer learning finished.");           

            // Evaluate the trained model on the passed test dataset.
            Console.WriteLine("Making bulk predictions and evaluating model's " +
               "quality...");

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {(elapsedMs / 1000)/60} minutes");

            context.Model.Save(model, trainImageDataShuffle.Schema, "model.zip");

            var valImages = LoadImagesFromDirectory(validationImageFolderPath, true);
            var valImageDataView = context.Data.LoadFromEnumerable(valImages);

            valImageDataView = TransformDataView(context, valImageDataView, validationImageFolderPath);

            var metric = context.MulticlassClassification.Evaluate(model.Transform(valImageDataView));

            Console.WriteLine($"LogLoss : " + metric.LogLoss + "\n" +
                $"MacroAccuracy : " + metric.MacroAccuracy + "\n" +
                $"MicroAccuracy : " + metric.MicroAccuracy + "\n");

            Console.ReadLine();
        }

        public static IDataView TransformDataView(MLContext context, IDataView dataView, string imageFolderPath)
        {
            return context.Transforms.Conversion
               .MapValueToKey("Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
               .Append(context.Transforms.LoadRawImageBytes("Image", imageFolderPath, "ImagePath"))
               .Fit(dataView)
               .Transform(dataView);
        }

        public static string DownloadImageSet(string imagesDownloadFolder, string datasetUrl, string datasetFullName)
        {
            // get a set of images to teach the network about the new classes
            Web.Download(datasetUrl, imagesDownloadFolder, datasetFullName);

            Compress.UnZip(Path.Join(imagesDownloadFolder, datasetFullName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(datasetFullName);
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder,bool useFolderNameAsLabel = true)
           => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
               .Select(x => new ImageData { ImagePath = x.imagePath, Label = x.label });
    }
}
