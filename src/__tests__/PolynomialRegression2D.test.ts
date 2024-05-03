import { describe, it, expect } from 'vitest';

import { PolynomialRegression2D } from '../PolynomialRegression2D';

describe('2D polinomial fit', () => {
  const X = new Array(21);
  const y = new Array(21);
  for (let i = 0; i < 21; ++i) {
    X[i] = [i, i + 10];
    y[i] = i + 20;
  }

  const pf = new PolynomialRegression2D(X, y, {
    order: 2,
  });

  it('Training coefficients', () => {
    const estimatedCoefficients = [
      1.5587e1, 3.8873e-1, 5.2582e-3, 4.8498e-1, 2.1127e-3, -7.3709e-3,
    ];
    for (let i = 0; i < estimatedCoefficients.length; ++i) {
      expect(pf.coefficients.get(i, 0)).toBeCloseTo(
        estimatedCoefficients[i],
        1e-2,
      );
    }
  });

  it('Prediction', () => {
    const test = new Array(11);
    let val = 0.5;
    for (let i = 0; i < 11; ++i) {
      test[i] = [val, val + 10];
      val++;
    }

    const y = pf.predict(test);

    let j = 0;
    for (let i = 20.5; i < 30.5; i++, j++) {
      expect(y[j]).toBeCloseTo(i, 1e-2);
    }
  });

  it('Other function test', () => {
    const testValues = [
      15.041667, 9.375, 5.041667, 2.041667, 0.375, 0.041667, 1.041667, 3.375,
      7.041667, 12.041667,
    ];

    const len = 21;

    const X = new Array(len);
    let val = 5.0;
    const y = new Array(len);
    for (let i = 0; i < len; ++i, val += 0.5) {
      X[i] = [val, val];
      y[i] = val * val + val * val;
    }

    const polynomialRegression2D = new PolynomialRegression2D(X, y, {
      order: 2,
    });

    const test = 10;
    let x1 = -4.75;
    let x2 = 4.75;
    const X1 = new Array(test);
    for (let i = 0; i < test; ++i) {
      X1[i] = [x1, x2];
      x1++;
      x2--;
    }

    const predict = polynomialRegression2D.predict(X1);
    for (let i = 0; i < testValues.length; ++i) {
      expect(predict[i]).toBeCloseTo(testValues[i], 1e-2);
    }
  });
});
