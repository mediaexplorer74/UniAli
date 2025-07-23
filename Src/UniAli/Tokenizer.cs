/*using System;
using System.Collections.Generic;
using System.Linq;

namespace UniAli
{
    public static class StringExtensions
    {
        public static IEnumerable<string> SplitAndKeep(this string s, string delimiters)
        {
            int start = 0, index;
            while ((index = s.IndexOfAny(delimiters.ToCharArray(), start)) != -1)
            {
                if (index - start > 0)
                    yield return s.Substring(start, index - start);
                yield return s.Substring(index, 1);
                start = index + 1;
            }
            if (start < s.Length)
                yield return s.Substring(start);
        }
    }

    public class Tokenizer
    {
        private readonly Dictionary<string, long> _vocab;
        private readonly long _maxSequenceLength;

        public Tokenizer(Dictionary<string, long> vocab, long maxSequenceLength)
        {
            _vocab = vocab;
            _maxSequenceLength = maxSequenceLength;
        }

        public long[] Encode(string input)
        {
            // Простая токенизация по пробелам и знакам препинания
            // В реальном приложении используйте нормальный токенизатор, например, из библиотеки
            var tokens = input.ToLower()
                             .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                             .SelectMany(s => s.SplitAndKeep(".,!?;:()[]{}'\""))
                             .Where(s => !string.IsNullOrWhiteSpace(s))
                             .Take((int)_maxSequenceLength)
                             .ToArray();

            var encodedTokens = new long[_maxSequenceLength];
            Array.Fill(encodedTokens, _vocab["[PAD]"]); // Заполняем паддингом

            for (long i = 0; i < Math.Min(tokens.Length, _maxSequenceLength); i++)
            {
                var token = tokens[i].Trim().ToString(); // check it
                if (_vocab.TryGetValue(token, out long index))
                {
                    encodedTokens[i] = index;
                }
                else
                {
                    // Пробуем найти токен без учета регистра
                    var lowerToken = token.ToLower();
                    var found = _vocab.FirstOrDefault(kv => kv.Key.ToLower() == lowerToken);
                    encodedTokens[i] = found.Value != 0 ? found.Value : _vocab["[UNK]"];
                }
            }

            return encodedTokens;
        }



        public string Decode(long[] encodedTokens)
        {
            var tokens = new List<string>();

            foreach (var token in encodedTokens)
            {
                if (token == 0) break; // padding token

                if (_vocab.ContainsValue(token))
                {
                    var key = _vocab.FirstOrDefault(x => x.Value == token).Key;
                    tokens.Add(key);
                }
                else
                {
                    tokens.Add("[UNK]"); // unknown token
                }
            }

            return string.Join(" ", tokens);
        }
    }
}
*/

using System.Collections.Generic;
using UniAli;

public class Tokenizer
{
    private readonly BertTokenizer _tokenizer;

    public Tokenizer(Dictionary<string, long> vocab, int maxSequenceLength)
    {
        _tokenizer = new BertTokenizer(vocab, maxSequenceLength);
    }

    public long[] Encode(string input)
    {
        return _tokenizer.Encode(input);
    }

    public string Decode(long[] encodedTokens)
    {
        return _tokenizer.Decode(encodedTokens);
    }
}