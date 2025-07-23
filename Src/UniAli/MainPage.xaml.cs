using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.ApplicationModel;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace UniAli
{

    public sealed partial class MainPage : Page
    {
        private InferenceSession _session;

        //private Tokenizer _tokenizer;
        private BertTokenizer _tokenizer;

        // MainPage
        public MainPage()
        {
            this.InitializeComponent();
            InitializeModel();
        }


        private async void InitializeModel()
        {
            try
            {
                // Путь к модели в папке Assets
                var modelPath = Path.Combine(Package.Current.InstalledLocation.Path, "Assets", "model_quantized.onnx");

                // Создаем сессию ONNX Runtime
                _session = new InferenceSession(modelPath);

                // Инициализация токенизатора (упрощенная версия)
                // В реальном приложении нужно загрузить токенизатор из файлов модели

                var vocab = new Dictionary<string, long>();

                /*vocab.Add("[PAD]", 100); // Padding token, used to fill sequences to a fixed length.
                vocab.Add("[UNK]", 101); // Unknown token, used for out-of-vocabulary words
                vocab.Add("[CLS]", 102); // Classification token, used as the first token for classification
                vocab.Add("[SEP]", 103); // Separator token, used to separate sequences or sentences
                vocab.Add("[MASK]", 104); // Mask token, used for masked language modeling tasks.
                vocab.Add("[BOS]", 105); // Beginning  of Sequence tokens, used to mark the start and end of a sequence.
                vocab.Add("[EOS]", 106); // End of Sequence tokens, used to mark the start and end of a sequence.
                for (int i = 107; i < 1000; i++)
                {
                    vocab.Add($"example{i}", i);
                }*/
                vocab.Add("[PAD]", 0);
                vocab.Add("[UNK]", 1);
                vocab.Add("[CLS]", 2);
                vocab.Add("[SEP]", 3);
                vocab.Add("[MASK]", 4);
                vocab.Add("[BOS]", 5);
                vocab.Add("[EOS]", 6);

                int maxLength = 2; // можно брать и какое-то иное число под определенные потребности

                //_tokenizer = new Tokenizer(vocab, maxLength);
                _tokenizer = new BertTokenizer(vocab, 512); // max sequence length

            }
            catch (Exception ex)
            {
                // Обработка ошибок инициализации
                System.Diagnostics.Debug.WriteLine($"Ошибка загрузки модели: {ex.Message}");
            }//
        }

        private string GenerateResponse(string input)
        {
            if (_session == null || _tokenizer == null)
                return "Модель не загружена";

            try
            {
                // Токенизация входного текста
                long[] tokens = _tokenizer.Encode(input);

                // Подготовка входных данных для модели (Создаем тензоры ONNX)
                //DenseTensor<long> inputTensor = new DenseTensor<long>(tokens, true);
                //DenseTensor<long> attentionMaskTensor = new DenseTensor<long>(tokens, true);

                // call: DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride = false)
                DenseTensor<long> inputTensor = new DenseTensor<long>(default, tokens.Select(t => (int)t).ToArray(), false);
                DenseTensor<long> attentionMaskTensor = new DenseTensor<long>(default, tokens.Select(t => (int)t).ToArray(), false);
                //long[,] attentionMask = new long[0, tokens.Length];


                // Создаем входные данные для модели
                List<NamedOnnxValue> inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
                };

                using (var results = _session.Run(inputs))
                {
                    // Получение выходных данных
                    Tensor<float> output = results.FirstOrDefault(r => r.Name == "logits")?.AsTensor<float>();
                    if (output != null)
                    {
                        // Декодирование результата (упрощенная версия)
                        long[] responseTokens = new long[] { 100, 200, 300 }; // Пример
                        return _tokenizer.Decode(responseTokens);
                    }
                }              
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Ошибка генерации ответа: {ex.Message}");
            }

            return "Извините, произошла ошибка при генерации ответа";
        }//


        private void SendButton_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrWhiteSpace(InputTextBox.Text))
            {
                // Показываем вопрос пользователя
                ResponseTextBlock.Text = $"Вы: {InputTextBox.Text}\n\n";

                // Генерируем ответ
                var response = GenerateResponse(InputTextBox.Text);

                // Показываем ответ модели
                ResponseTextBlock.Text += $"UniAli: {response}";

                // Очищаем поле ввода
                InputTextBox.Text = string.Empty;
            }//
        }

        private void InputTextBox_KeyDown(object sender, KeyRoutedEventArgs e)
        {
            if (e.Key == Windows.System.VirtualKey.Enter)
            {
                SendButton_Click(sender, e);
            }
        }//

    }
}
