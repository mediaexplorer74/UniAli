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
        public bool IsInputEnabled => !IsGenerating;
        private bool IsGenerating = false;

        private readonly List<(string, bool)> _messages = new List<(string, bool)>();
        private TikToken _tokenizer;
        private CanvasTextLayout _textLayout;

        public MainPage()
        {
            this.InitializeComponent();
            Loaded += MainPage_Loaded;
        }

        private async void MainPage_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                _tokenizer = TikToken.EncodingForModel("gpt-3.5-turbo");
                _messages.Add(("Привет! Я Qwen1.5 работаю на твоём крутом девайсе. Спроси что-нибудь!", false));
                RedrawChat();
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                _messages.Add(($"Ошибка: {ex.Message}", false));
                RedrawChat();
            }
        }

        private async void SendButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(InputBox.Text))
                return;

            var userMessage = InputBox.Text;
            InputBox.Text = "";
            _messages.Add((userMessage, true));
            RedrawChat();

            IsGenerating = true;
            Bindings.Update();

            try
            {
                var response = await GenerateResponseAsync(userMessage);
                _messages.Add((response, false));
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
                _messages.Add(($"Ошибка генерации: {ex.Message}", false));
            }
            finally
            {
                IsGenerating = false;
                Bindings.Update();
                RedrawChat();
            }
        }

        private async Task<string> GenerateResponseAsync(string prompt)
        {
            var session = App.Session;
            if (session == null) throw new InvalidOperationException("Модель не загружена");

            // Токенизация с форматом Qwen
            var promptTokens = new List<int> { 151644, 8948, 198 }; // <|im_start|>user
            promptTokens.AddRange(_tokenizer.Encode(prompt));
            promptTokens.AddRange(new[] { 151645, 198 }); // <|im_end|>\n
            promptTokens.AddRange(new[] { 151644, 77091, 198 }); // <|im_start|>assistant

            var inputTensor = new DenseTensor<long>(
                new[] { 1, promptTokens.Count },
                false//new[] { 1, promptTokens.Count }
             );

            for (int i = 0; i < promptTokens.Count; i++)
            {
                inputTensor[0, i] = promptTokens[i];
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(new[] { 1, promptTokens.Count }, false)),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(new[] { 1, promptTokens.Count }, false)),
                //NamedOnnxValue.CreateFromTensor("past_key_values", new DenseTensor<float>(new[] { 1, 0, 0, 0, 0, 0 }, false)) // Пустой past_key_values
                NamedOnnxValue.CreateFromTensor("past_key_values.0.key", new DenseTensor<long>(new[] { 1, promptTokens.Count }, false)),
            };

            // Конфиг генерации
            const int maxLength = 100;
            var outputTokens = new List<long>();

            for (int i = 0; i < maxLength; i++)
            {
                using (var results = session.Run(inputs))
                {    
                   var logits = results.First().AsTensor<float>();

                    // Жадное декодирование
                    long nextToken = ArgMax(logits, inputTensor.Dimensions[1] - 1);

                    // Конец генерации
                    if (nextToken == 151645) break; // <|im_end|>

                    outputTokens.Add(nextToken);

                    // Обновляем входные данные
                    var newDims = new[] { 1, inputTensor.Dimensions[1] + 1 };
                    var newInputTensor = new DenseTensor<long>(newDims);

                    for (int j = 0; j < inputTensor.Length; j++)
                    {
                        //newInputTensor.Buffer[j] = inputTensor.Buffer[j];
                        newInputTensor.Buffer.Span[j] = inputTensor.Buffer.Span[j];
                    }

                    //newInputTensor.Buffer[inputTensor.Length] = nextToken;
                    newInputTensor.Buffer.Span[(int)inputTensor.Length] = nextToken;
                    inputTensor = newInputTensor;

                    inputs[0] = NamedOnnxValue.CreateFromTensor("input_ids", inputTensor);
               }
            }

            //return _tokenizer.Decode(outputTokens.ToArray());
            //return _tokenizer.Decode(outputTokens.Select(x => (int)x).ToArray());
            return _tokenizer.Decode(outputTokens.Select(x => (int)x).ToList());
        }

        private static long ArgMax(Tensor<float> logits, int lastIndex)
        {
            float[] logitsArray = logits.ToArray();
            var slice = logitsArray.Skip(lastIndex * logits.Dimensions[2]).Take(logits.Dimensions[2]).ToArray();

            float max = float.MinValue;
            long maxIndex = 0;

            for (int i = 0; i < slice.Length; i++)
            {
                if (slice[i] > max)
                {
                    max = slice[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private void RedrawChat() => ChatCanvas.Invalidate();

        private void ChatCanvas_Draw(CanvasControl sender, CanvasDrawEventArgs args)
        {
            using (var session = args.DrawingSession)
            {

                session.Clear(Colors.Transparent);

                var y = 10f;
                var textFormat = new CanvasTextFormat
                {
                    FontSize = 14,
                    WordWrapping = CanvasWordWrapping.Wrap
                };

                foreach (var (msg, isUser) in _messages)
                {
                    var color = isUser ? Colors.DodgerBlue : Colors.Green;
                    var layout = new CanvasTextLayout(session, msg, textFormat,
                        (float)(sender.ActualWidth - 20), 0);

                    session.DrawTextLayout(layout, 10, y, color);
                    y += (float)layout.LayoutBounds.Height + 10;
                }

                _textLayout?.Dispose();
                _textLayout = null;
            }
        }
    }
}