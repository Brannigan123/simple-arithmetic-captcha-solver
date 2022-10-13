import { createCaptcha } from 'svg-captcha-fixed';
import isSvg from 'is-svg';



function randomNumber(min, max) {
    return ~~(Math.random() * (max - min + 1) + min);
}

function createMathExpr() {
    const digit1 = randomNumber(1, 9);
    const digit2 = randomNumber(1, 9);
    const text = `${digit1} + ${digit2}`;
    const data = createCaptcha(text, { background: '#ffffff', color: false, noise: 3 });
    return { digit1, digit2, text, data };
};

function generate() {
    while (true) {
        var {digit1, digit2, text, data } = createMathExpr();
        if (isSvg(data)) {
            return { text, data }
        }
    }
}

process.stdout.write(JSON.stringify(generate()))