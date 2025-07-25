﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using TiktokenSharp.Model;
using TiktokenSharp.Utils;

namespace TiktokenSharp.Services
{
    internal class EncodingManager
    {

        private static readonly Lazy<EncodingManager> _instance =
            new Lazy<EncodingManager>(() => new EncodingManager());

        public static EncodingManager Instance
        {
            get { return _instance.Value; }
        }

        public string PBEFileDirectory { get; set; } = Path.Combine(AppContext.BaseDirectory, "bpe");


        const string ENDOFTEXT = "<|endoftext|>";
        const string FIM_PREFIX = "<|fim_prefix|>";
        const string FIM_MIDDLE = "<|fim_middle|>";
        const string FIM_SUFFIX = "<|fim_suffix|>";
        const string ENDOFPROMPT = "<|endofprompt|>";

        static Dictionary<string, string> MODEL_TO_ENCODING = new Dictionary<string, string>()
                                                            {
                                                            // chat
                                                            { "gpt-4", "cl100k_base" },
                                                            { "gpt-3.5-turbo", "cl100k_base" },
                                                            { "gpt-3.5-turbo-16k", "cl100k_base" },

                                                            // text
                                                            { "text-davinci-003", "p50k_base" },
                                                            { "text-davinci-002", "p50k_base" },
                                                            { "text-davinci-001", "r50k_base" },
                                                            { "text-curie-001", "r50k_base" },
                                                            { "text-babbage-001", "r50k_base" },
                                                            { "text-ada-001", "r50k_base" },
                                                            { "davinci", "r50k_base" },
                                                            { "curie", "r50k_base" },
                                                            { "babbage", "r50k_base" },
                                                            { "ada", "r50k_base" },
                                                            // code
                                                            { "code-davinci-002", "p50k_base" },
                                                            { "code-davinci-001", "p50k_base" },
                                                            { "code-cushman-002", "p50k_base" },
                                                            { "code-cushman-001", "p50k_base" },
                                                            { "davinci-codex", "p50k_base" },
                                                            { "cushman-codex", "p50k_base" },
                                                            // edit
                                                            { "text-davinci-edit-001", "p50k_edit" },
                                                            { "code-davinci-edit-001", "p50k_edit" },
                                                            // embeddings
                                                            { "text-embedding-ada-002", "cl100k_base" },
                                                            // old embeddings
                                                            { "text-similarity-davinci-001", "r50k_base" },
                                                            { "text-similarity-curie-001", "r50k_base" },
                                                            { "text-similarity-babbage-001", "r50k_base" },
                                                            { "text-similarity-ada-001", "r50k_base" },
                                                            { "text-search-davinci-doc-001", "r50k_base" },
                                                            { "text-search-curie-doc-001", "r50k_base" },
                                                            { "text-search-babbage-doc-001", "r50k_base" },
                                                            { "text-search-ada-doc-001", "r50k_base" },
                                                            { "code-search-babbage-code-001", "r50k_base" },
                                                            { "code-search-ada-code-001", "r50k_base" },
                                                            // open source
                                                            { "gpt2", "gpt2" }
                                                        };
        EncodingManager()
        {

        }

        /// <summary>
        /// Get encoding setting with model name.
        /// </summary>
        /// <param name="modelName">gpt-4 gpt-3.5-turbo ...</param>
        /// <returns></returns>
        public EncodingSettingModel GetEncodingSetting(string modelOrEncodingName)
        {
            var encodingName = MODEL_TO_ENCODING.FirstOrDefault(a => modelOrEncodingName.StartsWith(a.Key)).Value;

            if (string.IsNullOrEmpty(encodingName))
            {
                encodingName = MODEL_TO_ENCODING.FirstOrDefault(a => a.Key.StartsWith(modelOrEncodingName)).Value;
            }

            if (string.IsNullOrEmpty(encodingName))
            {
                if (MODEL_TO_ENCODING.Any(a => a.Value == modelOrEncodingName))
                {
                    //modelOrEncodingName is encoding name
                    encodingName = modelOrEncodingName;
                }
            }

            return GetEncoding(encodingName);
        }

        /// <summary>
        /// Get encoding setting with model name.
        /// </summary>
        /// <param name="modelName">gpt-4 gpt-3.5-turbo ...</param>
        /// <returns></returns>
        public async Task<EncodingSettingModel> GetEncodingSettingAsync(string modelOrEncodingName)
        {
            var encodingName = MODEL_TO_ENCODING.FirstOrDefault(a => a.Key.StartsWith(modelOrEncodingName)).Value;

            if (string.IsNullOrEmpty(encodingName))
            {
                if (MODEL_TO_ENCODING.Any(a => a.Value == modelOrEncodingName))
                {
                    //modelOrEncodingName is encoding name
                    encodingName = modelOrEncodingName;
                }
            }

            return await GetEncodingAsync(encodingName);
        }

        



        /// <summary>
        /// Get encoding setting with encoding name.
        /// </summary>
        /// <param name="encodingName">cl100k_base p50k_base ...</param>
        /// <returns></returns>
        public EncodingSettingModel GetEncoding(string encodingName)
        {
            if (!string.IsNullOrEmpty(encodingName))
            {
                switch (encodingName)
                {
                    case "gpt2":
                        {
                            //TODO
                            throw new NotImplementedException();
                        }
                    case "r50k_base":
                        {
                            //TODO
                            throw new NotImplementedException(); ;
                        }
                    case "p50k_base":
                        {
                            return p50k_base().Result;
                        }
                    case "p50k_edit":
                        {
                            //TODO
                            throw new NotImplementedException();
                        }
                    case "cl100k_base":
                        {
                            return cl100k_base().Result;
                        }
                    default:
                        throw new NotImplementedException();
                }

            }
            else
            {
                Debug.WriteLine("[error] TiktokenSharp - GetEncoding ex.: " + "Unsupported model " + encodingName);
                throw new NotImplementedException("Unsupported model");
            }
        }





        /// <summary>
        /// Get encoding setting with encoding name.
        /// </summary>
        /// <param name="encodingName">cl100k_base p50k_base ...</param>
        /// <returns></returns>
        public async Task<EncodingSettingModel> GetEncodingAsync(string encodingName)
        {
            if (!string.IsNullOrEmpty(encodingName))
            {
                switch (encodingName)
                {
                    case "gpt2":
                        {
                            //TODO
                            throw new NotImplementedException();
                        }
                    case "r50k_base":
                        {
                            //TODO
                            throw new NotImplementedException(); ;
                        }
                    case "p50k_base":
                        {
                            return await p50k_base();
                        }
                    case "p50k_edit":
                        {
                            //TODO
                            throw new NotImplementedException();
                        }
                    case "cl100k_base":
                        {
                            return await cl100k_base();
                        }
                    default:
                        throw new NotImplementedException();
                }

            }
            else
            {
                Debug.WriteLine("[error] TiktokenSharp - GetEncodingAsync ex.: " + "Unsupported model "+ encodingName);
                throw new NotImplementedException("Unsupported model");
            }
        }



        private Dictionary<byte[], int> LoadTikTokenBpeFromLocal(string tikTokenBpeFile)
        {
            var contents = File.ReadAllLines(tikTokenBpeFile, Encoding.UTF8);
            var bpeDict = new Dictionary<byte[], int>(contents.Length, new ByteArrayComparer());

            foreach (var line in contents.Where(l => !string.IsNullOrWhiteSpace(l)))
            {
                var tokens = line.Split();
                var tokenBytes = Convert.FromBase64String(tokens[0]);
                var rank = int.Parse(tokens[1]);
                bpeDict.Add(tokenBytes, rank);
            }

            return bpeDict;
        }

        private async Task<Dictionary<byte[], int>> LoadTikTokenBpe(string tikTokenBpeFile, string filename)
        {
            string localFilePath;
            /*if (tikTokenBpeFile.StartsWith("http"))
            {
                var fileName = Path.GetFileName(tikTokenBpeFile);
                var saveDir = PBEFileDirectory; //Path.Combine(AppContext.BaseDirectory, "bpe");

                / * try
                {
                    //If an exception occurs, it means that the folder cannot be created. Change the storage folder to the Temp directory.
                    if (!Directory.Exists(saveDir))
                    {
                        Directory.CreateDirectory(saveDir);
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("[ex] TiktokenSharp - LoadTikTokenBpe ex.: " + ex.Message);
                    saveDir = Path.GetTempPath();
                } * /
                //dirty hack
                saveDir = Path.GetTempPath();


                localFilePath = Path.Combine(saveDir, fileName);
                if (!File.Exists(localFilePath))
                {
                    using (var client = new HttpClient())
                    {
                        //client.DownloadFile(tikTokenBpeFile, localFilePath);
                        //File.WriteAllBytes(localFilePath, data);
                        var data = await client.GetByteArrayAsync(tikTokenBpeFile);
                        File.WriteAllBytes(localFilePath, data);
                    }
                }
            }
            else
            {
                localFilePath = tikTokenBpeFile;
            }*/

            //Dirty hack
            localFilePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            localFilePath = Path.Combine(localFilePath, /*"p50k_base.tiktoken"*/filename);

            var bpeDict = new Dictionary<byte[], int>(new ByteArrayComparer());

            try
            {
                var lines = File.ReadAllLines(localFilePath, Encoding.UTF8);
                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    var tokens = line.Split(' ');
                    if (tokens.Length != 2)
                    {
                        throw new FormatException($"Invalid file format: {localFilePath}");
                    }

                    var tokenBytes = Convert.FromBase64String(tokens[0]);
                    var rank = int.Parse(tokens[1]);
                    bpeDict[tokenBytes] = rank;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ex] Failed to load TikTokenBpe from {localFilePath}: {ex.Message}", ex);
                throw new InvalidOperationException($"Failed to load TikTokenBpe from {localFilePath}: {ex.Message}", ex);
            }

            return bpeDict;
        }


        private async Task<EncodingSettingModel> cl100k_base()
        {
            //When using the mod for the first time, the pbe file will be downloaded over the network.
            Dictionary<byte[], int> mergeable_ranks = await LoadTikTokenBpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", 
                "cl100k_base.tiktoken");
            Dictionary<string, int> special_tokens = new Dictionary<string, int>{
                                    { ENDOFTEXT, 100257},
                                    { FIM_PREFIX, 100258},
                                    { FIM_MIDDLE, 100259},
                                    { FIM_SUFFIX, 100260},
                                    { ENDOFPROMPT, 100276}
                                };

            return new EncodingSettingModel()
            {
                Name = "cl100k_base",
                PatStr = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                MergeableRanks = mergeable_ranks,
                SpecialTokens = special_tokens
            };
        }


        private async Task<EncodingSettingModel> p50k_base()
        {
            //When using the mod for the first time, the pbe file will be downloaded over the network.
            var mergeable_ranks = await LoadTikTokenBpe("https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken", 
                "p50k_base.tiktoken");
            var special_tokens = new Dictionary<string, int>{
                                    { ENDOFTEXT, 50256}
                                };

            return new EncodingSettingModel()
            {
                Name = "p50k_base",
                ExplicitNVocab = 50281,
                PatStr = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                MergeableRanks = mergeable_ranks,
                SpecialTokens = special_tokens
            };
        }
    }
}
