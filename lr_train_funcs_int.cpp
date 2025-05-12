//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2023, Duality Technologies Inc.
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#include "lr_train_funcs_int.h"
#include "pt_matrix.h"
#include "utils/debug.h"
#include "enc_matrix.h"
#include "math.h"

// Copied from pt_matrix.h rewritten to use int
void MatrixScalarMultInt(MatInt &A, prim_type t)
{
  for (usint i = 0; i < A.size(); i++)
  {
    for (usint j = 0; j < A[i].size(); j++)
    {
      A[i][j] = t * A[i][j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////
// Observe that if we pass in the scalingFactor (e.g lr / numRows) we can save on a multiplication
MatInt InitializeLogReg(MatInt &X, MatInt &y, float scalingFactor)
{
  /////////////////////////////////////////
  // update this for our problem
  /////////////////////////////////////////

  if (X.size() <= 0)
  {
    std::cerr << "Please provide a data matrix with positive number of rows." << std::endl;
    exit(0);
  }

#ifdef ENABLE_DEBUG
  std::cerr << "Initialization - Input data X (showing only 5 rows): " << std::endl;
  Mat Xsub = Mat(X.begin(), X.begin() + 5);
  PrintMatrix(Xsub);
  std::cerr << std::endl;
#endif // ENABLE_DEBUG

  // Compute X transpose
  // note X tranpose is the same CT packing as x Just labeled differntly since
  // X mat_col_major == X' mat_row_major
  // copy XT = X
  MatInt XT = MatInt(X.begin(), X.end());

  // take negative of XT
  MatrixScalarMultInt(XT, -1.0 * scalingFactor);

#ifdef ENABLE_DEBUG
  std::cerr << "Initialization - X transpose (showing only 5 rows, 5 columns): " << std::endl;
  Mat XTsub = Mat(5);
  for (usint i = 0; i < XTsub.size(); i++)
    XTsub[i] = Vec(XT[i].begin(), XT[i].begin() + 5);
  PrintMatrix(XTsub);
  std::cerr << std::endl;
#endif // ENABLE_DEBUG
  return (XT);
}
///////////////////////////////////////////////////////////////////////////////////////
// Functions needed for the LR Gradient calculation

// Replacement for EvalLogistic
// Maclaurin coefficients for 1/(1+e^{-x})
// coefficents from https://www.emathhelp.net/calculators/calculus-1/taylor-and-maclaurin-series-calculator/?f=1%2F%281%2Be%5E%28-x%29%29&p=0&n=7&v=
CT EvalLogisticTaylor(CC &cc, const KeyPair &keys, CT &ctLogits, int scale, int batchSize)
{
  // Horner's method 2nd degree free
  // seems to be more accurate + needs lower ptm
  int64_t coeff0 = 0.5 * scale  * scale;
  int64_t coeff1 = 0.25 * scale;
  
  PT pt_init = cc->MakePackedPlaintext(VecInt(batchSize, 0));
  PT ptCoeff1 = cc->MakePackedPlaintext(VecInt(batchSize, coeff1));
  PT ptCoeff0 = cc->MakePackedPlaintext(VecInt(batchSize, coeff0));

  CT ct_poly = cc->Encrypt(keys.publicKey, pt_init);
  ct_poly = cc->EvalMult(ct_poly, ctLogits);

  // Nothing to add coeff2 is 0
  ct_poly = cc->EvalMult(ct_poly, ctLogits);

  ct_poly = cc->EvalAdd(ct_poly, ptCoeff1);
  ct_poly = cc->EvalMult(ct_poly, ctLogits);

  ct_poly = cc->EvalAdd(ct_poly, ptCoeff0);

  return ct_poly;
}

// Replacement for MatrixVectorProductRow
void MatrixVectorProductRowBFV(CC &cc, const KeyPair &keys, const CT &ctX, CT &ctThetas, usint rowSize, CT &ctLogits)
{
  // TODO check if this holds always
  ctLogits = cc->EvalInnerProduct(ctX, ctThetas, rowSize);
}

/*
https://openfhe.discourse.group/t/is-there-any-example-about-how-to-use-evalsumrows-evalsumcols/687/2?
     sum col
1111 4
2222 4
3 sum row
*/

void MatrixVectorProductColBFV(CC &cc, const KeyPair &keys, const CT &cMat, 
  const CT &cVecColCloned, int64_t rowSize, int64_t numRows,
    CT &cProduct)
{
  // EvalSumRows step by step
  //   * Sums all elements over column-vectors in a matrix

  // Generate key for every rotation we will do 
  // need to only do ceil(log2(numRows)) rotations
  // switches so every element i gets added to each other
  std::vector<int> rotations;
    for (usint i = 1; i < numRows; i *= 2) {
        rotations.push_back(i * rowSize);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);

  // Multiply matrix by cloned column vector
  auto ct_mult = cc->EvalMult(cMat, cVecColCloned);

  // Sum across rows via rotations + additions
  CT ct_out = ct_mult;

  for (auto &it: rotations)
  {
    auto ct_rot = cc->EvalRotate(ct_out, it);
    ct_out      = cc->EvalAdd(ct_out, ct_rot);
  }

  cProduct = ct_out;
}

// Will scale down the ciphertext by scale
CT ReEncryptScaleDown(CC &cc, CT &in, KeyPair keys, int scale, int batchSize)
{
  PT plain;
  cc->Decrypt(keys.secretKey, in, &plain);
  VecInt raw = plain->GetPackedValue();

  for (auto &x : raw)
    x = x / scale;

  plain = cc->MakePackedPlaintext(raw);
  return cc->Encrypt(keys.publicKey, plain);
}

///////////////////////////////////////////////////////////////////////////////////////
void EncLogRegCalculateGradient(
    CC &cc,
    const CT &ctX,
    const CT &ctNegXt,
    const CT &ctLabels,
    CT &ctThetas,
    CT &ctGradStoreInto,
    const usint rowSize,
    const usint numRows,
    const KeyPair &keys,
    bool debug,
    int scale,
    int debugPlaintextLength)
{
  OPENFHE_DEBUG_FLAG(false);
  // We use the same notation as in
  //    https://eprint.iacr.org/2018/662.pdf
  //    It seems like their labels are {-1, 1} which we do not use. Change accordingly
  CT ctLogits;
  PT dbg;

  size_t batchSize = cc->GetCryptoParameters()
                         ->GetEncodingParams()
                         ->GetBatchSize();

  if (debug)
  {
    cc->Decrypt(keys.secretKey, ctThetas, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tDEBUG: Thetas: " << dbg;
    cc->Decrypt(keys.secretKey, ctX, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tDEBUG: Xs: " << dbg;
  }

  // Line 4
  MatrixVectorProductRowBFV(cc, keys, ctX, ctThetas, rowSize, ctLogits);

  ctLogits = ReEncryptScaleDown(cc, ctLogits, keys, scale, batchSize);

  if (debug)
  {
    cc->Decrypt(keys.secretKey, ctLogits, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\t Logits: " << dbg;
    std::cout << std::endl;
  }

  // Line 5/6
  auto preds = EvalLogisticTaylor(cc, keys, ctLogits, scale, batchSize);

  preds = ReEncryptScaleDown(cc, preds, keys, scale, batchSize);

  if (debug)
  {
    cc->Decrypt(keys.secretKey, preds, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\t preds: " << dbg;
    std::cout << std::endl;
  }

  // Line 8 - see Page 9 for their notation
  OPENFHE_DEBUG("\tPre-Residual");
  auto residual = cc->EvalSub(ctLabels, preds);

  if (debug)
  {
    cc->Decrypt(keys.secretKey, residual, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\t residual: " << dbg;
    std::cout << std::endl;
  }

  // TODO this 64 is hardcoded should be automatically set
  MatrixVectorProductColBFV(cc, keys, ctNegXt, residual, rowSize, numRows, ctGradStoreInto);
  ctGradStoreInto = ReEncryptScaleDown(cc, ctGradStoreInto, keys, scale, batchSize);

  if (debug)
  {
    cc->Decrypt(keys.secretKey, ctGradStoreInto, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\t scaled gradient: " << dbg;
    std::cout << std::endl;
  }
}

///////////////////////////////////////////////////////////////
void BoundCheckMat(const Mat &inMat, const double bound)
{

  usint numRows = inMat.size();
  usint numCols = inMat[0].size();

  // yes this is slow...
  for (usint i = 0; i < numRows; i++)
  {
    for (usint j = 0; j < numCols; j++)
    {
      if (abs((int)inMat[i][j]) >= (int)bound)
      {
        std::cout << "element at [" << i << "," << j << "] is " << inMat[i][j] << " bounds " << bound << std::endl;
      }
    }
  }
}

////////////////////////////////const//////////////////////////////
PT ReEncrypt(CC &cc, CT &ctx, const KeyPair &keys)
{

  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("In ReEncrypt");
  // reencrypt x
  PT xPT;
  OPENFHE_DEBUG("Decrypt");
  cc->Decrypt(keys.secretKey, ctx, &xPT);

  VecInt x = xPT->GetPackedValue();

  xPT = cc->MakePackedPlaintext(x);

  OPENFHE_DEBUG("Encrypt() ");
  ctx = cc->Encrypt(keys.publicKey, xPT);
  return xPT; // return this for debug purposes...
}

int ReturnDepth(const CT &ct)
{
  auto mulDepth = ct->GetElements()[0].GetNumOfElements() - 1;
  auto scaling = ct->GetScalingFactor();
  std::cout << "mult Depth: " << mulDepth << " Scaling: " << scaling << std::endl;
  return (mulDepth);
}

double ComputeLoss(const Mat &b, const Mat &X, const Mat &y)
{
  // Based off of https://stackoverflow.com/a/47798689/18031872
  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("In ComputeLoss");
  usint numSamp = X.size(); // n_samp

  /////////////////////////////////////////////////////////////////
  // Calculate t1: matmul(-y.T, log(yHat)
  /////////////////////////////////////////////////////////////////
  // yHat = sigmoid(X * beta);
  Mat yHat = Mat(numSamp, Vec(1, 0.0));
  MatrixMult(X, b, yHat);
  MatrixSigmoid(yHat);
  // log(yHat)
  Mat logYHat = Mat(numSamp, Vec(1, 0.0));
  MatrixLog(yHat, logYHat);

  Mat yT = Mat(y[0].size(), Vec(y.size(), 0.0));
  MatrixTransp(y, yT);
  MatrixScalarMult(yT, -1);
  Mat t1Mat = Mat(1, Vec(1, 0.0));
  MatrixMult(yT, logYHat, t1Mat);
  // PrintMatrix(t1Mat);

  /////////////////////////////////////////////////////////////////
  // t2: matmult(
  //     t2_a,
  //     t2_b
  //     )
  //  t2_a = 1 - y.T
  //  t2_b = log(1 - yHat)
  /////////////////////////////////////////////////////////////////
  // from earlier it exists as -yT. We change it back here
  // so we can do a sub. Less confusing for newer readers
  Mat t2Mat_a = Mat(yT.size(), Vec(yT[0].size(), 0.0));
  MatrixScalarMult(yT, -1);
  // Getting t2_a
  ScalarSubMat(1, yT, t2Mat_a);
  OPENFHE_DEBUG("Got t2_a: 1-yT");

  Mat t2Mat_b = Mat(y.size(), Vec(1, 0.0));
  ScalarSubMat(1, yHat, t2Mat_b);
  MatrixLog(t2Mat_b, t2Mat_b);
  OPENFHE_DEBUG("Got t2_b: log(1-yHat)");

  Mat t2Mat = Mat(1, Vec(1, 0.0));
  MatrixMult(t2Mat_a, t2Mat_b, t2Mat);

  // Should now have a Mat Scalar that we add up
  Mat loglikelihood = Mat(1, Vec(1, 0.0));
  MatrixMatrixSub(t1Mat, t2Mat, loglikelihood);
  return loglikelihood[0][0] / double(numSamp);
}
