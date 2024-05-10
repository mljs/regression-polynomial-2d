import { describe, it, expect } from 'vitest';

import { PolynomialRegression2D } from '../PolynomialRegression2D';

describe('2D polinomial fit', () => {
  const x = new Array(21);
  const y = new Array(21);
  const z = new Array(21);
  for (let i = 0; i < 21; ++i) {
    x[i] = i;
    y[i] = i + 10;
    z[i] = i + 20;
  }

  const pf = new PolynomialRegression2D({ x, y }, z, {
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
    const xTest = new Float64Array(11);
    const yTest = new Float64Array(11);
    let val = 0.5;
    for (let i = 0; i < 11; ++i) {
      xTest[i] = val;
      yTest[i] = val + 10;
      val++;
    }
    const y = pf.predict({ x: xTest, y: yTest });

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

    const x = new Array(len);
    let val = 5.0;
    const y = new Array(len);
    const z = new Array(len);
    for (let i = 0; i < len; ++i, val += 0.5) {
      x[i] = val;
      y[i] = val;
      z[i] = val * val + val * val;
    }

    const polynomialRegression2D = new PolynomialRegression2D({ x, y }, z, {
      order: 2,
    });

    const length = 10;
    let x1 = -4.75;
    let x2 = 4.75;
    const testData = {
      x: Float64Array.from({ length }),
      y: Float64Array.from({ length }),
    };
    for (let i = 0; i < length; ++i) {
      testData.x[i] = x1;
      testData.y[i] = x2;
      x1++;
      x2--;
    }

    const predict = polynomialRegression2D.predict(testData);
    for (let i = 0; i < testValues.length; ++i) {
      expect(predict[i]).toBeCloseTo(testValues[i], 1e-2);
    }
  });
});
it('Other function test', () => {
  const len = 4;

  const x = new Array(len);
  let val = 5.0;
  const y = new Array(len);
  const z = new Array(len);
  for (let i = 0; i < len; ++i, val += 0.5) {
    x[i] = val;
    y[i] = val;
    z[i] = val * val + val * val;
  }

  expect(() => new PolynomialRegression2D({ x, y }, z, { order: 2 })).toThrow(
    'Insufficient number of points to create regression model.',
  );
});
