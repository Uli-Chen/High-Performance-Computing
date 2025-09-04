#pragma once

// Host API: Matrix Addition
// C = A + B
// All matrices are in row-major order, size: w x h
void MatAdd(const float* A, const float* B, float* C, int w, int h);
