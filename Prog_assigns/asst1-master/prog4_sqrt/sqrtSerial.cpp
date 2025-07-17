#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

static __m256 calculate_errors(__m256 guesses, __m256 xes) {
        // error = fabs(guess * guess * x - 1.f);
	const static __m256 ones = _mm256_set1_ps(1.0f);
	const static __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
	__m256 errors = _mm256_mul_ps(guesses, guesses);
    	errors = _mm256_mul_ps(xes, errors);
	errors = _mm256_sub_ps(errors, ones);
	errors = _mm256_and_ps(errors, sign_mask);
	return errors;
}

static __m256 calculate_guesses(__m256 guesses, __m256 xes) {
        // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
	const static __m256 three = _mm256_set1_ps(3.0f);
	const static __m256 half_one = _mm256_set1_ps(0.5f);
	__m256 original = guesses;
	guesses = _mm256_mul_ps(original, three);
	__m256 tmp = _mm256_mul_ps(xes, original);
	tmp = _mm256_mul_ps(tmp, original);
	tmp = _mm256_mul_ps(tmp, original);
	guesses = _mm256_sub_ps(guesses, tmp);
	guesses = _mm256_mul_ps(guesses, half_one);
	return guesses;
}

void sqrtAVX2(int N, float initialGuess, float values[], float output[]) {
    __m256 k_thresholds = _mm256_set1_ps(0.00001f);
    for (int i = 0; i < N; i += 8) {
	    // load
	    __m256 xes = _mm256_loadu_ps(&values[i]);
	    __m256 guesses = _mm256_set1_ps(initialGuess);
	    __m256 errors = calculate_errors(guesses, xes);
	    // calculate
	    while (true) { 
		    __m256 cmp_ret = _mm256_cmp_ps(errors, k_thresholds,_CMP_NGT_UQ);
		    int cmp = _mm256_movemask_ps(cmp_ret);
		    if (cmp == 0xFF) break;
		    guesses = calculate_guesses(guesses, xes);
		    errors = calculate_errors(guesses, xes);
	    }

	    xes = _mm256_mul_ps(xes, guesses);
	    // store
	    _mm256_storeu_ps(&output[i], xes);
    }
}
