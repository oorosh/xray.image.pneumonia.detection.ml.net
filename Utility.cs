using System;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.IO.Compression;
using Microsoft.ML.Transforms;
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using Microsoft.ML;
using System.IO;
using System.Linq;

namespace pneumonia.detection
{
    public static class Utility
    {
        public static IDataView GetTrainImageData(string trainImageFolderPath, MLContext context)
        {
            var trainImages = LoadImagesFromDirectory(trainImageFolderPath, true);
            var trainImageData = context.Data.LoadFromEnumerable(trainImages);
            var trainImageDataShuffle = context.Data.ShuffleRows(trainImageData);
            return trainImageDataShuffle;
        }

        public static IDataView TransformDataView(MLContext context, IDataView dataView, string imageFolderPath)
        {
            // Apply transforms to the input dataset:
            // MapValueToKey : map 'string' type labels to keys
            // LoadImages : load raw images to "Image" column
            return context.Transforms.Conversion
               .MapValueToKey("Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
               .Append(context.Transforms.LoadRawImageBytes("Image", imageFolderPath, "ImagePath"))
               .Fit(dataView)
               .Transform(dataView);
        }

        public static string DownloadImageSet(string imagesDownloadFolder, string datasetUrl, string datasetFullName)
        {
            // get a set of images to teach the network about the new classes
            Download(datasetUrl, imagesDownloadFolder, datasetFullName);

            UnZip(Path.Join(imagesDownloadFolder, datasetFullName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(datasetFullName);
        }        

        public static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
            {
                Console.WriteLine($"{relativeFilePath} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {relativeFilePath}");
            var download = Task.Run(() => wc.DownloadFile(url, relativeFilePath));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {relativeFilePath}");

            return true;
        }

        public static void ExtractGZip(string gzipFileName, string targetDir)
        {
            // Use a 4K buffer. Any larger is a waste.    
            byte[] dataBuffer = new byte[4096];

            using (System.IO.Stream fs = new FileStream(gzipFileName, FileMode.Open, FileAccess.Read))
            {
                using (GZipInputStream gzipStream = new GZipInputStream(fs))
                {
                    // Change this to your needs
                    string fnOut = Path.Combine(targetDir, Path.GetFileNameWithoutExtension(gzipFileName));

                    using (FileStream fsOut = File.Create(fnOut))
                    {
                        StreamUtils.Copy(gzipStream, fsOut, dataBuffer);
                    }
                }
            }
        }

        public static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static void ExtractTGZ(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                using (var inStream = File.OpenRead(gzArchiveName))
                {
                    using (var gzipStream = new GZipInputStream(inStream))
                    {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
           => LoadImagesFromFileSystem(folder, useFolderNameAsLabel)
               .Select(x => new ImageData { ImagePath = x.imagePath, Label = x.label });

        public static IEnumerable<(string imagePath, string label)> LoadImagesFromFileSystem(string folder,bool useFolderNameasLabel)
        {
            var imagesPath = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            return useFolderNameasLabel
                ? imagesPath.Select(imagePath => (imagePath, Directory.GetParent(imagePath).Name))
                : imagesPath.Select(imagePath =>
                {
                    var label = Path.GetFileName(imagePath);
                    for (var index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                    return (imagePath, label);
                });
        }
    }
}
