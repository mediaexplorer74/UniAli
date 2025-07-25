using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Text;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using TiktokenSharp;
using Windows.UI;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;

namespace UniAli
{
    public sealed partial class MainPage : Page
    {
        // Константы из config.json
        private const int NumLayers = 28;
        private const int NumKeyValueHeads = 8;
        private const int HeadDim = 128;
        private const int VocabSize = 151936;
        private const int BosTokenId = 151643;
        private const int EosTokenId = 151645;
        private const int MaxPositionEmbeddings = 40960;

        public bool IsInputEnabled => !IsGenerating && !IsLoadingModel;
        public bool IsGenerating { get; private set; }
        public bool IsLoadingModel { get; private set; } = true;

        private List<(string Text, bool IsUser)> _messages = new List<(string, bool)>();
        private TikToken _tokenizer;
        private readonly object _syncLock = new object();

        public MainPage()
        {
            this.InitializeComponent();
            this.Loaded += MainPage_Loaded;
        }

        private async void MainPage_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                await Task.Run(() => {
                    _tokenizer = TikToken.EncodingForModel("gpt-3.5-turbo");
                });

                _messages.Add(("Система инициализирована. Qwen3-0.6B готова к работе!", false));
                _messages.Add(($"Параметры модели: слои={NumLayers}, heads={NumKeyValueHeads}, dim={HeadDim}", false));
                IsLoadingModel = false;
                Bindings.Update();
                RedrawChat();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Ошибка инициализации токенизатора: {ex}");
                _messages.Add(("Ошибка загрузки токенизатора", false));
                IsLoadingModel = false;
                Bindings.Update();
                RedrawChat();
            }
        }

        private async void SendButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(InputBox.Text)) return;

            var userMessage = InputBox.Text;
            InputBox.Text = "";
            _messages.Add((userMessage, true));
            RedrawChat();

            IsGenerating = true;
            Bindings.Update();

            try
            {
                var maxTokens = (int)MaxTokensSlider.Value;
                var mode = GenerationMode.SelectedIndex;
                var stopwatch = Stopwatch.StartNew();

                string response = await Task.Run(() =>
                    GenerateResponse(userMessage, maxTokens, mode));

                _messages.Add(($"{response}\n[Генерация заняла: {stopwatch.Elapsed.TotalSeconds:F2} сек]", false));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Ошибка генерации: {ex}");
                _messages.Add(($"Ошибка генерации: {ex.Message}", false));
            }
            finally
            {
                IsGenerating = false;
                Bindings.Update();
                RedrawChat();
                GC.Collect();
            }
        }

        /*private string GenerateResponse(string prompt, int maxTokens, int mode)
        {
            lock (_syncLock)
            {
                var session = App.Session;
                if (session == null)
                {
                    Debug.WriteLine("Ошибка: сессия ONNX не инициализирована");
                    return "Модель не загружена";
                }

                try
                {
                    // Токенизация промпта
                    List<long> promptTokens = new List<long> { BosTokenId, 8948, 198 };
                    promptTokens.AddRange(_tokenizer.Encode(prompt).Select(x => (long)x));
                    promptTokens.AddRange(new long[] { EosTokenId, 198 });
                    promptTokens.AddRange(new long[] { BosTokenId, 77091, 198 });

                    // Подготовка входных тензоров
                    int[] inputShape = new int[] { 1, promptTokens.Count };
                    DenseTensor<long> inputIds = new DenseTensor<long>(inputShape);
                    DenseTensor<long> attentionMask = new DenseTensor<long>(inputShape);
                    DenseTensor<long> positionIds = new DenseTensor<long>(inputShape);

                    for (int i = 0; i < promptTokens.Count; i++)
                    {
                        inputIds[0, i] = promptTokens[i];
                        attentionMask[0, i] = 1;
                        positionIds[0, i] = i;
                    }

                    // Подготовка начальных past_key_values
                    var pastKeyValues = new List<NamedOnnxValue>();
                    var kvShape = new int[] { 1, NumKeyValueHeads, 0, HeadDim };

                    for (int i = 0; i < NumLayers; i++)
                    {
                        var keyTensor = new DenseTensor<Float16>(kvShape);
                        var valueTensor = new DenseTensor<Float16>(kvShape);

                        pastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                            $"past_key_values.{i}.key", keyTensor));

                        pastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                            $"past_key_values.{i}.value", valueTensor));
                    }

                    // Генерация ответа
                    var outputTokens = new List<long>();
                    var currentInput = inputIds;
                    var currentAttention = attentionMask;
                    var currentPositions = positionIds;

                    for (int step = 0; step < maxTokens; step++)
                    {
                        int currentLength = currentPositions.Dimensions[1];
                        if (currentLength >= MaxPositionEmbeddings)
                        {
                            Debug.WriteLine("Достигнута максимальная длина контекста");
                            break;
                        }

                        Debug.WriteLine($"Шаг генерации {step}:");
                        Debug.WriteLine($"  Input IDs: {string.Join(" , ", currentInput.Dimensions.ToArray())}");
                        Debug.WriteLine($"  Attention: {string.Join(" , ", currentAttention.Dimensions.ToArray())}");
                        Debug.WriteLine($"  Positions: {string.Join(" , ", currentPositions.Dimensions.ToArray())}");

                        var inputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("input_ids", currentInput),
                            NamedOnnxValue.CreateFromTensor("attention_mask", currentAttention),
                            NamedOnnxValue.CreateFromTensor("position_ids", currentPositions)
                        };
                        inputs.AddRange(pastKeyValues);

                        using var results = session.Run(inputs);
                        var logits = results.First().AsTensor<float>();
                        var nextToken = SampleToken(logits, mode);

                        if (nextToken == EosTokenId)
                        {
                            Debug.WriteLine("Обнаружен токен конца последовательности");
                            break;
                        }

                        outputTokens.Add(nextToken);
                        pastKeyValues = UpdatePastKeyValues(results);

                        currentInput = new DenseTensor<long>(new int[] { 1, 1 });
                        currentInput[0, 0] = nextToken;

                        int newAttentionLength = currentAttention.Dimensions[1] + 1;
                        currentAttention = new DenseTensor<long>(new int[] { 1, newAttentionLength });
                        for (int j = 0; j < newAttentionLength; j++)
                            currentAttention[0, j] = 1;

                        currentPositions = new DenseTensor<long>(new int[] { 1, 1 });
                        currentPositions[0, 0] = currentLength;
                    }

                    List<int> outputTokensList = outputTokens.Select(x => (int)x).ToList();
                    string decodedText = _tokenizer.Decode(outputTokensList);
                    return decodedText;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Ошибка в процессе генерации: {ex}");
                    return $"Ошибка генерации: {ex.Message}";
                }
            }
        }*/
        private string GenerateResponse(string prompt, int maxTokens, int mode)
        {
            lock (_syncLock)
            {
                var session = App.Session;
                if (session == null)
                {
                    Debug.WriteLine("Ошибка: сессия ONNX не инициализирована");
                    return "Модель не загружена";
                }

                try
                {
                    // Токенизация промпта
                    List<long> promptTokens = new List<long> { BosTokenId, 8948, 198 };
                    promptTokens.AddRange(_tokenizer.Encode(prompt).Select(x => (long)x));
                    promptTokens.AddRange(new long[] { EosTokenId, 198 });
                    promptTokens.AddRange(new long[] { BosTokenId, 77091, 198 });

                    Debug.WriteLine($"Токены после токенизации: {string.Join(", ", promptTokens)}");

                    // Подготовка входных тензоров
                    int[] inputShape = new int[] { 1, promptTokens.Count };
                    DenseTensor<long> inputIds = new DenseTensor<long>(inputShape);
                    DenseTensor<long> attentionMask = new DenseTensor<long>(inputShape);
                    DenseTensor<long> positionIds = new DenseTensor<long>(inputShape);

                    for (int i = 0; i < promptTokens.Count; i++)
                    {
                        inputIds[0, i] = promptTokens[i];
                        attentionMask[0, i] = 1;
                        positionIds[0, i] = i;
                    }

                    // Подготовка начальных past_key_values
                    var pastKeyValues = new List<NamedOnnxValue>();
                    var kvShape = new int[] { 1, NumKeyValueHeads, 0, HeadDim };

                    for (int i = 0; i < NumLayers; i++)
                    {
                        var keyTensor = new DenseTensor<Float16>(kvShape);
                        var valueTensor = new DenseTensor<Float16>(kvShape);

                        pastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                            $"past_key_values.{i}.key", keyTensor));

                        pastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                            $"past_key_values.{i}.value", valueTensor));
                    }

                    // Генерация ответа
                    var outputTokens = new List<long>();
                    var currentInput = inputIds;
                    var currentAttention = attentionMask;
                    var currentPositions = positionIds;

                    for (int step = 0; step < maxTokens; step++)
                    {
                        int currentLength = currentPositions.Dimensions[1];
                        if (currentLength >= MaxPositionEmbeddings)
                        {
                            Debug.WriteLine("Достигнута максимальная длина контекста");
                            break;
                        }

                        Debug.WriteLine($"Шаг генерации {step}:");
                        Debug.WriteLine($"  Input IDs: {string.Join(", ", currentInput.Dimensions.ToArray())}");
                        Debug.WriteLine($"  Attention: {string.Join(", ", currentAttention.Dimensions.ToArray())}");
                        Debug.WriteLine($"  Positions: {string.Join(", ", currentPositions.Dimensions.ToArray())}");

                        var inputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("input_ids", currentInput),
                            NamedOnnxValue.CreateFromTensor("attention_mask", currentAttention),
                            NamedOnnxValue.CreateFromTensor("position_ids", currentPositions)
                        };
                        inputs.AddRange(pastKeyValues);

                        using (var results = session.Run(inputs))
                        {
                            var logits = results.First().AsTensor<float>();
                            var nextToken = SampleToken(logits, mode);

                            if (nextToken == EosTokenId)
                            {
                                Debug.WriteLine("Обнаружен токен конца последовательности");
                                break;
                            }

                            outputTokens.Add(nextToken);
                            pastKeyValues = UpdatePastKeyValues(results);


                            currentInput = new DenseTensor<long>(new int[] { 1, 1 });
                            currentInput[0, 0] = nextToken;
                        }

                        int newAttentionLength = currentAttention.Dimensions[1] + 1;
                        currentAttention = new DenseTensor<long>(new int[] { 1, newAttentionLength });
                        for (int j = 0; j < newAttentionLength; j++)
                            currentAttention[0, j] = 1;

                        currentPositions = new DenseTensor<long>(new int[] { 1, 1 });
                        currentPositions[0, 0] = currentLength;
                    }

                    Debug.WriteLine($"Выходные токены: {string.Join(", ", outputTokens)}");

                    List<int> outputTokensList = outputTokens.Select(x => (int)x).ToList();
                    string decodedText = _tokenizer.Decode(outputTokensList);
                    Debug.WriteLine($"Декодированный текст: {decodedText}");
                    return decodedText;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Ошибка в процессе генерации: {ex}");
                    return $"Ошибка генерации: {ex.Message}";
                }
            }
        }

        private List<NamedOnnxValue> UpdatePastKeyValues(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            var newPastKeyValues = new List<NamedOnnxValue>();

            for (int i = 0; i < NumLayers; i++)
            {
                var keyName = $"present.{i}.key";
                var valueName = $"present.{i}.value";

                var presentKey = results.FirstOrDefault(r => r.Name == keyName);
                var presentValue = results.FirstOrDefault(r => r.Name == valueName);

                if (presentKey != null && presentValue != null)
                {
                    // Используем float вместо Float16
                    //newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                    //    $"past_key_values.{i}.key", presentKey.AsTensor<float>()));

                    // newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                    //    $"past_key_values.{i}.value", presentValue.AsTensor<float>()));
                    newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                    $"past_key_values.{i}.key", presentKey.AsTensor<Microsoft.ML.OnnxRuntime.Float16>()));

                    newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                        $"past_key_values.{i}.value", presentValue.AsTensor<Microsoft.ML.OnnxRuntime.Float16>()));
                }
                else
                {
                    var emptyTensor = new DenseTensor<float>(new int[] { 1, NumKeyValueHeads, 0, HeadDim });

                    newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                        $"past_key_values.{i}.key", emptyTensor));

                    newPastKeyValues.Add(NamedOnnxValue.CreateFromTensor(
                        $"past_key_values.{i}.value", emptyTensor));

                    Debug.WriteLine($"Не найдены present значения для слоя {i}");
                }
            }

            return newPastKeyValues;
        }

        private long SampleToken(Tensor<float> logits, int mode)
        {
            float temperature = 0.6f;
            float topP = 0.9f;

            if (mode == 1)
            {
                temperature = 0.9f;
                topP = 0.95f;
            }
            else if (mode == 2)
            {
                temperature = 0.3f;
                topP = 0.5f;
            }

            var lastPos = logits.Dimensions[1] - 1;
            var logitsArray = new float[logits.Dimensions[2]];

            for (int i = 0; i < logitsArray.Length; i++)
                logitsArray[i] = logits[0, lastPos, i];

            for (int i = 0; i < logitsArray.Length; i++)
                logitsArray[i] /= temperature;

            if (topP < 1.0f)
            {
                ApplyTopPSampling(logitsArray, topP);
            }

            var maxLogit = logitsArray.Max();
            var expValues = logitsArray.Select(v => Math.Exp(v - maxLogit)).ToArray();
            var sumExp = expValues.Sum();
            var probs = expValues.Select(v => v / sumExp).ToArray();

            var random = new Random();
            var randValue = random.NextDouble();
            var cumulative = 0.0;

            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (randValue < cumulative)
                {
                    return i;
                }
            }

            return probs.Length - 1;
        }

        private void ApplyTopPSampling(float[] logits, float topP)
        {
            var sortedIndices = logits
                .Select((value, index) => new { Value = value, Index = index })
                .OrderByDescending(item => item.Value)
                .Select(item => item.Index)
                .ToArray();

            var sortedLogits = sortedIndices.Select(i => logits[i]).ToArray();

            var cumulativeProbs = new float[sortedLogits.Length];
            float sumExp = sortedLogits.Select(l => (float)Math.Exp(l)).Sum();
            if (sumExp == 0) sumExp = 1e-9f;

            cumulativeProbs[0] = (float)Math.Exp(sortedLogits[0]) / sumExp;

            for (int i = 1; i < sortedLogits.Length; i++)
            {
                cumulativeProbs[i] = cumulativeProbs[i - 1] + (float)Math.Exp(sortedLogits[i]) / sumExp;
            }

            int cutoffIndex = 0;
            for (int i = 0; i < cumulativeProbs.Length; i++)
            {
                if (cumulativeProbs[i] > topP)
                {
                    cutoffIndex = i;
                    break;
                }
            }

            for (int i = cutoffIndex; i < sortedIndices.Length; i++)
            {
                logits[sortedIndices[i]] = float.NegativeInfinity;
            }
        }

        private void RedrawChat()
        {
            ChatCanvas?.Invalidate();
        }

        private void ChatCanvas_Draw(CanvasControl sender, CanvasDrawEventArgs args)
        {
            lock (_syncLock)
            {
                using (var session = args.DrawingSession)
                {
                    session.Clear(Colors.Transparent);

                    float y = 10f;
                    var textFormat = new CanvasTextFormat
                    {
                        FontSize = 14,
                        WordWrapping = CanvasWordWrapping.Wrap
                    };

                    foreach (var (text, isUser) in _messages)
                    {
                        var color = isUser ? Colors.DodgerBlue : Colors.ForestGreen;
                        var layout = new CanvasTextLayout(session, text, textFormat,
                            (float)sender.ActualWidth - 40, 0);

                        session.DrawTextLayout(layout, 20, y, color);
                        y += (float)layout.LayoutBounds.Height + 15;
                    }
                }
            }
        }
    }
}