import { type NumberArray } from 'cheminfo-types';
import { Matrix, SVD } from 'ml-matrix';

import BaseRegression2D from './BaseRegression2D';

export interface PolynomialRegression2DOptions {
  /**
   *degree of the polynomial regression.
   * @default 2
   */
  order?: number;
}

// Implements the Kernel ridge regression algorithm.
// http://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf
export class PolynomialRegression2D extends BaseRegression2D {
  order: number;
  coefficients: Matrix;
  /**
   * Constructor for the 2D polynomial fitting
   *
   * @param inputs - independent or explanatory variable
   * @param outputs - dependent or response variable`
   * @constructor
   */
  constructor(
    inputs: NumberArray[],
    outputs: NumberArray[],
    options: PolynomialRegression2DOptions = {},
  ) {
    super();
    // @ts-expect-error internal use only
    if (inputs === true) {
      // @ts-expect-error internal use only
      this.coefficients = Matrix.columnVector(outputs.coefficients);
      // @ts-expect-error internal use only
      this.order = outputs.order;
      // @ts-expect-error internal use only
      if (outputs.r) {
        // @ts-expect-error internal use only
        this.r = outputs.r;
        // @ts-expect-error internal use only
        this.r2 = outputs.r2;
      }
      // @ts-expect-error internal use only
      if (outputs.chi2) {
        // @ts-expect-error internal use only
        this.chi2 = outputs.chi2;
      }
    } else {
      const { order = 2 } = options;
      this.order = order;
      this.coefficients = train(inputs, outputs, order);
    }
  }

  _predict(newInputs: NumberArray) {
    const x1 = newInputs[0];
    const x2 = newInputs[1];

    let y = 0;
    let column = 0;

    for (let i = 0; i <= this.order; i++) {
      for (let j = 0; j <= this.order - i; j++) {
        y += x1 ** i * x2 ** j * this.coefficients.get(column, 0);
        column++;
      }
    }

    return y;
  }

  toJSON() {
    return {
      name: 'polyfit2D',
      order: this.order,
      coefficients: this.coefficients,
    };
  }

  static load(json: Record<string, any>) {
    if (json.name !== 'polyfit2D') {
      throw new TypeError('not a polyfit2D model');
    }
    //@ts-expect-error internal use only
    return new PolynomialRegression2D(true, json);
  }
}

/**
 * Function that given a column vector return this: vector^power
 *
 * @param x - Column vector.
 * @param power - Pow number.
 * @return {Matrix}
 */
function powColVector(x: Matrix, power: number) {
  const result = x.clone();
  for (let i = 0; i < x.rows; ++i) {
    result.set(i, 0, result.get(i, 0) ** power);
  }
  return result;
}

/**
 * Function that fits the model given the data(x) and predictions(y).
 * The third argument is an object with the following options:
 * * order: order of the polynomial to fit.
 *
 * @param x - A matrix with n rows and 2 columns.
 * @param y - A vector of the prediction values.
 */
function train(
  x: NumberArray[] | Matrix,
  y: NumberArray[] | Matrix,
  order: number,
) {
  if (!Matrix.isMatrix(x)) x = new Matrix(x);
  //@ts-expect-error it is a internal error in matrix;
  if (!Matrix.isMatrix(y)) y = Matrix.columnVector(y);

  if (y.rows !== x.rows) {
    y = y.transpose();
  }

  if (x.columns !== 2) {
    throw new RangeError(
      `You give x with ${x.columns} columns and it must be 2`,
    );
  }
  if (x.rows !== y.rows) {
    throw new RangeError('x and y must have the same rows');
  }

  const examples = x.rows;
  const nbCoefficients = ((order + 2) * (order + 1)) / 2;

  const x1 = x.getColumnVector(0);
  const x2 = x.getColumnVector(1);

  const scaleX1 = 1.0 / x1.clone().abs().max();
  const scaleX2 = 1.0 / x2.clone().abs().max();
  const scaleY = 1.0 / y.clone().abs().max();

  x1.mulColumn(0, scaleX1);
  x2.mulColumn(0, scaleX2);
  y.mulColumn(0, scaleY);

  const A = new Matrix(examples, nbCoefficients);
  let col = 0;

  for (let i = 0; i <= order; ++i) {
    const limit = order - i;
    for (let j = 0; j <= limit; ++j) {
      const result = powColVector(x1, i).mulColumnVector(powColVector(x2, j));
      A.setColumn(col, result);
      col++;
    }
  }

  const svd = new SVD(A.transpose(), {
    computeLeftSingularVectors: true,
    computeRightSingularVectors: true,
    autoTranspose: false,
  });

  let qqs = Matrix.rowVector(svd.diagonal);
  qqs = qqs.apply((i, j) => {
    if (qqs.get(i, j) >= 1e-15) qqs.set(i, j, 1 / qqs.get(i, j));
    else qqs.set(i, j, 0);
  });

  const qqs1 = Matrix.zeros(examples, nbCoefficients);
  for (let i = 0; i < nbCoefficients; ++i) {
    qqs1.set(i, i, qqs.get(0, i));
  }

  qqs = qqs1;

  const U = svd.rightSingularVectors;
  const V = svd.leftSingularVectors;

  const coefficients = V.mmul(qqs.transpose()).mmul(U.transpose()).mmul(y);
  for (let i = 0, col = 0; i <= nbCoefficients; ++i) {
    const limit = order - i;
    for (let j = 0; j <= limit; ++j) {
      coefficients.set(
        col,
        0,
        (coefficients.get(col, 0) * scaleX1 ** i * scaleX2 ** j) / scaleY,
      );
      col++;
    }
  }

  return coefficients;
}
