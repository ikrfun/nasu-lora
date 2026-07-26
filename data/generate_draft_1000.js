const fs = require('fs');

const source = fs.readFileSync('data/train.jsonl', 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));

const questionFrames = [
  (q) => `${q} 初心者向けに教えて。`,
  (q) => `${q} 重要なポイントを三つに絞って。`,
  (q) => `${q} 失敗しないための注意点は？`,
  (q) => `${q} 具体例を一つ入れて説明して。`,
  (q) => `${q} 短く、結論から答えて。`,
  (q) => `${q} メリットと注意点を分けて教えて。`,
  (q) => `${q} 実際に行動するときの手順を教えて。`,
  (q) => `${q} 那須・ナス・NASのどれかに少し寄り道して答えて。`,
  (q) => `${q} 子どもにも分かるように説明して。`,
  (q) => `${q} よくある勘違いも添えて。`,
  (q) => `${q} 旅行や日常で役立つ形にして。`,
  (q) => `${q} 一言で答えたあと、補足を一つだけ付けて。`,
];

const answerFrames = [
  (a) => a,
  (a) => `結論から言うと、${a}`,
  (a) => `${a} まずは無理なく、小さく試すのがよいなす。`,
  (a) => `${a} 条件や最新情報は、実行前に公式案内を確認するなす🍆`,
  (a) => `${a} 迷ったら、目的・安全・復元方法の順に確認すると整理しやすいなす。`,
];

const records = [];
let id = 0;
for (let seedIndex = 0; records.length < 1000; seedIndex += 1) {
  const seedIndexInSource = seedIndex % source.length;
  const seed = source[seedIndexInSource];
  const cycleSuffix = seedIndex >= source.length ? ' 別の切り口でも考えて。' : '';
  const originalQuestion = seed.messages[0].content;
  const originalAnswer = seed.messages[1].content;
  for (let frameIndex = 0; frameIndex < questionFrames.length && records.length < 1000; frameIndex += 1) {
    const question = questionFrames[frameIndex](originalQuestion) + cycleSuffix;
    const answer = answerFrames[(seedIndex + frameIndex) % answerFrames.length](originalAnswer);
    records.push({
      draft_id: `draft-${String(id + 1).padStart(4, '0')}`,
      source_seed: seedIndexInSource,
      variant: frameIndex,
      review_status: 'needs_human_review',
      messages: [
        { role: 'user', content: question },
        { role: 'assistant', content: answer },
      ],
    });
    id += 1;
  }
}

if (records.length !== 1000) {
  throw new Error(`expected 1000 draft records, got ${records.length}`);
}

fs.writeFileSync(
  'data/draft_1000.jsonl',
  records.map((record) => JSON.stringify(record)).join('\n') + '\n',
);
console.log(`wrote ${records.length} records to data/draft_1000.jsonl`);
