use std::collections::{BinaryHeap, HashMap};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Улучшенный BPE токенизатор
#[derive(Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// token string -> id
    vocab: HashMap<String, usize>,
    /// id -> token string
    ivocab: Vec<String>,
    /// merge rules: (left_id, right_id) -> merged_id
    #[serde(
        serialize_with = "serialize_merges",
        deserialize_with = "deserialize_merges"
    )]
    merges: HashMap<(usize, usize), usize>,

    // Специальные токены
    pub pad_token: usize,
    pub bos_token: usize,
    pub eos_token: usize,
    pub unk_token: usize,
}

fn serialize_merges<S: Serializer>(
    merges: &HashMap<(usize, usize), usize>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let vec: Vec<((usize, usize), usize)> = merges.iter().map(|(k, v)| (*k, *v)).collect();
    vec.serialize(s)
}

fn deserialize_merges<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<HashMap<(usize, usize), usize>, D::Error> {
    let vec: Vec<((usize, usize), usize)> = Vec::deserialize(d)?;
    Ok(vec.into_iter().collect())
}

impl BpeTokenizer {
    /// Создаёт новый токенизатор
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut ivocab = Vec::new();

        let specials = vec!["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, &tok) in specials.iter().enumerate() {
            vocab.insert(tok.to_string(), i);
            ivocab.push(tok.to_string());
        }

        BpeTokenizer {
            vocab,
            ivocab,
            merges: HashMap::new(),
            pad_token: 0,
            bos_token: 1,
            eos_token: 2,
            unk_token: 3,
        }
    }

    /// Обучение BPE
    pub fn train<I, S>(&mut self, sentences: I, target_vocab_size: usize)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sequences: Vec<Vec<String>> = Vec::new();
        let mut char_counts: HashMap<String, usize> = HashMap::new();

        println!("Подсчёт символов и подготовка последовательностей...");

        for s in sentences {
            let txt = s.as_ref().trim();
            if txt.is_empty() {
                continue;
            }

            // Простая пре-токенизация (лучше, чем просто по символам)
            let mut tokens = Vec::new();
            for word in txt.split_whitespace() {
                if word.is_empty() {
                    continue;
                }
                // Добавляем префикс Ġ для начала слова (как в GPT)
                let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                if !chars.is_empty() {
                    chars[0] = format!("Ġ{}", chars[0]);
                }
                tokens.extend(chars);
                tokens.push(" ".to_string()); // разделитель между словами
            }

            for tok in &tokens {
                *char_counts.entry(tok.clone()).or_insert(0) += 1;
            }
            sequences.push(tokens);
        }

        // Добавляем все уникальные символы в словарь
        for (ch, _) in char_counts {
            if !self.vocab.contains_key(&ch) {
                let id = self.ivocab.len();
                self.vocab.insert(ch.clone(), id);
                self.ivocab.push(ch);
            }
        }

        let start_size = self.vocab.len();
        println!("Начальный размер словаря: {}", start_size);
        println!("Начинаем слияния до размера {}", target_vocab_size);

        // Основной цикл BPE
        while self.vocab.len() < target_vocab_size {
            if (self.vocab.len() - start_size) % 300 == 0 {
                println!(
                    "  Merge progress: {}/{} (vocab size: {})",
                    self.vocab.len() - start_size,
                    target_vocab_size - start_size,
                    self.vocab.len()
                );
            }

            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();

            for seq in &sequences {
                for window in seq.windows(2) {
                    if let (Some(&a), Some(&b)) =
                        (self.vocab.get(&window[0]), self.vocab.get(&window[1]))
                    {
                        *pair_counts.entry((a, b)).or_insert(0) += 1;
                    }
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Находим самую частую пару
            let (&best_pair, _) = pair_counts.iter().max_by_key(|&(_, &count)| count).unwrap();

            let left = &self.ivocab[best_pair.0];
            let right = &self.ivocab[best_pair.1];
            let merged = if right.starts_with('Ġ') {
                format!("{}{}", left, right.trim_start_matches('Ġ'))
            } else {
                format!("{}{}", left, right)
            };

            let merged_id = if let Some(&id) = self.vocab.get(&merged) {
                id
            } else {
                let id = self.ivocab.len();
                self.vocab.insert(merged.clone(), id);
                self.ivocab.push(merged.clone());
                id
            };

            self.merges.insert(best_pair, merged_id);

            // Применяем слияние
            for seq in &mut sequences {
                let mut i = 0;
                while i < seq.len() - 1 {
                    let a_id = *self.vocab.get(&seq[i]).unwrap();
                    let b_id = *self.vocab.get(&seq[i + 1]).unwrap();

                    if (a_id, b_id) == best_pair {
                        seq[i] = merged.clone();
                        seq.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        println!(
            "Токенизатор обучен! Финальный размер словаря: {}",
            self.vocab.len()
        );
    }

    /// Кодирование текста в токены
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unk = self.unk_token;
        let mut ids: Vec<usize> = text
            .chars()
            .map(|c| c.to_string())
            .map(|t| *self.vocab.get(&t).unwrap_or(&unk))
            .collect();

        if ids.len() <= 1 {
            return ids;
        }

        // Используем твою эффективную реализацию с BinaryHeap
        #[derive(Eq, PartialEq)]
        struct Candidate {
            priority: usize,
            pos: usize,
        }

        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .priority
                    .cmp(&self.priority)
                    .then_with(|| self.pos.cmp(&other.pos))
            }
        }

        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let n = ids.len();
        let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let mut next: Vec<Option<usize>> = (0..n)
            .map(|i| if i + 1 < n { Some(i + 1) } else { None })
            .collect();
        let mut active = vec![true; n];
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

        for i in 0..n - 1 {
            let pair = (ids[i], ids[i + 1]);
            if let Some(&merged) = self.merges.get(&pair) {
                heap.push(Candidate {
                    priority: merged,
                    pos: i,
                });
            }
        }

        while let Some(Candidate { pos, .. }) = heap.pop() {
            if !active[pos] {
                continue;
            }
            let right = match next[pos] {
                Some(r) if active[r] => r,
                _ => continue,
            };

            let pair = (ids[pos], ids[right]);
            let merged = match self.merges.get(&pair) {
                Some(&m) => m,
                None => continue,
            };

            ids[pos] = merged;
            active[right] = false;
            next[pos] = next[right];
            if let Some(nr) = next[right] {
                prev[nr] = Some(pos);
            }

            if let Some(l) = prev[pos] {
                let pair = (ids[l], ids[pos]);
                if let Some(&m) = self.merges.get(&pair) {
                    heap.push(Candidate {
                        priority: m,
                        pos: l,
                    });
                }
            }
            if let Some(r) = next[pos] {
                let pair = (ids[pos], ids[r]);
                if let Some(&m) = self.merges.get(&pair) {
                    heap.push(Candidate { priority: m, pos });
                }
            }
        }

        let mut result = Vec::new();
        let mut cursor = (0..n).find(|&i| active[i]);
        while let Some(pos) = cursor {
            result.push(ids[pos]);
            cursor = next[pos];
        }
        result
    }

    /// Декодирование токенов обратно в текст
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut text = ids
            .iter()
            .filter_map(|&id| self.ivocab.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join("");

        // Чистим Ġ и лишние пробелы
        text = text.replace("Ġ", " ");
        text = text.replace("  ", " ");
        text.trim().to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
