using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace UniAli
{
    public class BertTokenizer
    {
        private readonly Dictionary<string, long> _vocab;
        private readonly long _maxSequenceLength;

        public BertTokenizer(Dictionary<string, long> vocab, long maxSequenceLength)
        {
            _vocab = vocab;
            _maxSequenceLength = maxSequenceLength;
        }

        public long[] Encode(string input)
        {
            var tokens = Tokenize(input);
            var encodedTokens = new long[_maxSequenceLength];
            Array.Fill(encodedTokens, 0); // padding token

            for (long i = 0; i < Math.Min(tokens.Length, _maxSequenceLength); i++)
            {
                var token = tokens[i];
                if (_vocab.TryGetValue(token, out long index))
                {
                    encodedTokens[i] = index;
                }
                else
                {
                    // Use the [UNK] token for unknown words
                    encodedTokens[i] = _vocab["[UNK]"];
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

        private string[] Tokenize(string input)
        {
            // WordPiece tokenization algorithm
            var tokens = new List<string>();
            var word = "";

            foreach (var c in input)
            {
                if (char.IsLetterOrDigit(c))
                {
                    word += c;
                }
                else
                {
                    if (word.Length > 0)
                    {
                        tokens.Add(word);
                        word = "";
                    }

                    tokens.Add(c.ToString());
                }
            }

            if (word.Length > 0)
            {
                tokens.Add(word);
            }

            return tokens.ToArray();
        }
    }
}