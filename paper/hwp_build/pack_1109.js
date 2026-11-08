// 원본 HWP를 읽어 스트림만 바꾼 새 파일을 쓴다 (원본은 수정하지 않음).
// node pack_1109.js <원본.hwp> <build_1109 출력 폴더/> <새 파일.hwp>
// cfb 1.2.2의 seed_cfb를 비활성화해야 불필요한 Sh33tJ5 스트림이 생기지 않는다 (README 참고).
const CFB = require('cfb'); const fs = require('fs');
const [src, dir, out] = process.argv.slice(2);
if (src === out) throw new Error('원본과 출력 경로가 같다');
const cfb = CFB.read(fs.readFileSync(src), {type: 'buffer'});
const put = (p, file) => { const d = fs.readFileSync(dir + file); const e = CFB.find(cfb, p);
  if (e) { e.content = d; e.size = d.length; } else CFB.utils.cfb_add(cfb, p, d); };
put('/DocInfo', 'DocInfo.bin'); put('/BodyText/Section0', 'Section0.bin'); put('/BodyText/Section1', 'Section1.bin');
put('/BinData/BIN0002.png', 'BIN0002.png.bin'); put('/BinData/BIN0003.png', 'BIN0003.png.bin'); put('/BinData/BIN0004.png', 'BIN0004.png.bin');
fs.writeFileSync(out, CFB.write(cfb, {type: 'buffer', fileType: 'cfb'}));
console.log(cfb.FullPaths.filter(p => !p.endsWith('/')).join('\n'));
