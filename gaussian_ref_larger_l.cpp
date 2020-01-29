/* ****************************** *
 * Implemented by Raymond K. ZHAO *
 *                                *
 * Discrete Gaussian Sampler      *
 * ****************************** */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vcl/vectorclass.h"
#include "vcl/vectormath_exp.h"
#include "vcl/vectormath_trig.h"

#include "randombytes.h"
#include "fastrandombytes.h"
#include "cpucycles.h"

#define BOX_MULLER_BYTES (8 * 8)

/* Box-Muller sampler precision: 53 bits */
#define FP_PRECISION 53
static const double INT64_DOUBLE_FACTOR = 1.0 / (1LL << FP_PRECISION);
#define FP_MASK ((1LL << FP_PRECISION) - 1)

/* Parameters used by FACCT */
#define COMP_ENTRY_SIZE 10
#define EXP_MANTISSA_PRECISION 52
#define EXP_MANTISSA_MASK ((1LL << EXP_MANTISSA_PRECISION) - 1)
#define R_MANTISSA_PRECISION (EXP_MANTISSA_PRECISION + 1)
#define R_MANTISSA_MASK ((1LL << R_MANTISSA_PRECISION) - 1)
#define R_EXPONENT_L (8 * COMP_ENTRY_SIZE - R_MANTISSA_PRECISION)
#define DOUBLE_ONE (1023LL << 52)

#define NORM_BATCH 8 /* The AVX2 Box-Muller sampler returns 8 samples in batch */
#define DISCRETE_BYTES (1 + NORM_BATCH * COMP_ENTRY_SIZE)

static const double SIGMA = 1048576.0; /* sigma=1048576 in our benchmark */
static const double DISCRETE_NORMALISATION = 0.5 * M_SQRT1_2 * M_2_SQRTPI * (1.0 / SIGMA); /* 1/S=1/(sigma*sqrt(2*pi)) */
static const double SIGMA_INV = -1.0 / (2 * SIGMA * SIGMA); /* -1/sigma^2 */

#define T 1000
#define TOTAL 1024

static double norm[NORM_BATCH];
static uint64_t head = NORM_BATCH;

/* Box-Muller sampler */
static inline void box_muller()
{
	unsigned char r[BOX_MULLER_BYTES];
	Vec4q l1, l2;
	Vec4d r1, r2;
	Vec4d r2_sin, r2_cos;
	
	fastrandombytes(r, BOX_MULLER_BYTES);
	
	l1.load((uint64_t *)r);
	l2.load(((uint64_t *)r) + 4);
	
	r1 = to_double((l1 & FP_MASK) + 1) * INT64_DOUBLE_FACTOR;
	r2 = to_double((l2 & FP_MASK) + 1) * INT64_DOUBLE_FACTOR;
	
	r1 = sqrt(-2.0 * log(r1)) * SIGMA;
	r2 = 2.0 * M_PI * r2;
	
	r2_sin = sincos(&r2_cos, r2);
	
	r2_cos = r1 * r2_cos;
	r2_sin = r1 * r2_sin;
	
	r2_cos.store(norm);
	r2_sin.store(norm + 4);
}

/* FACCT rejection check */
static inline int64_t comp(const unsigned char *r, const double x)
{
	uint64_t res = *((uint64_t *)(&x));
	uint64_t res_mantissa, res_exponent;
	uint64_t r1, r2;
	uint64_t r_mantissa, r_exponent;
	
	res_mantissa = (res & EXP_MANTISSA_MASK) | (1LL << EXP_MANTISSA_PRECISION);
	res_exponent = R_EXPONENT_L - 1023 + 1 + (res >> EXP_MANTISSA_PRECISION); 
	
	r1 = *((uint64_t *)r);
	r2 = (uint64_t)(*((uint16_t *)(r + 8)));
	
	r_mantissa = r1 & R_MANTISSA_MASK;
	r_exponent = (r1 >> R_MANTISSA_PRECISION) | (r2 << (64 - R_MANTISSA_PRECISION));
	
	return (res == DOUBLE_ONE) || ((r_mantissa < res_mantissa) && (r_exponent < (1LL << res_exponent))); 	
}

/* We can merge branches in Algorithm 2 in the paper as follows: 
 * Let y<--N(0, sigma^2)
 * Let b<--U({0,1})
 * If (b==0):
 *     Let y_r=round(y)-1
 *     Let boolean cmp=(y<=0.5)
 * Else:
 *     Let y_r=round(y)+1
 *     Let boolean cmp=(y>=-0.5)
 * EndIf
 * If (cmp is true):
 *     Let r<--U([0,1))
 *     If (r<exp(-((y_r+c_F)^2-y^2)/2sigma^2):
 *         Return y_r
 *     Else:
 *         Restart
 *     EndIf
 * Else:
 *     Restart
 * EndIf
 */
static int64_t discrete_gaussian(const double center)
{
	unsigned char r[DISCRETE_BYTES]; 	
	
	double c, cr, rc; 

	double y, yr, rej; 

	uint64_t b, i;
	int64_t cmp1;
	
	cr = round(center);
	c = cr - center;
	
	rc = exp(c * c * SIGMA_INV) * DISCRETE_NORMALISATION;

	fastrandombytes(r, COMP_ENTRY_SIZE);

	if (comp(r, rc))
	{
		return cr;
	}
	
	while (true)
	{
		fastrandombytes(r, DISCRETE_BYTES);

		for (i = 0; i < 8; i++, head++)
		{
			if (head >= NORM_BATCH)
			{
				head = 0;
				box_muller();
			}

			y = norm[head];
			b = (r[DISCRETE_BYTES - 1] >> i) & 0x01;
			
			if (b)
			{
				yr = round(y) + 1.0;
				cmp1 = (y >= -0.5);
			}
			else
			{
				yr = round(y) - 1.0;				
				cmp1 = (y <= 0.5);
			}
			
			if (cmp1)
			{
				rej = exp((yr + c + y) * (yr + c - y) * SIGMA_INV);
				
				if (comp(r + i * COMP_ENTRY_SIZE, rej))
				{
					head++;
					return yr + cr;
				}
			}
		}
	}	
}

int main()
{
	unsigned char seed[32];
	uint64_t i, t;
	int64_t sample[TOTAL];
	double center[TOTAL];
	
	long long cycle1, cycle2;
	srand(time(NULL));	
for (t = 0; t < T; t++)
{
	randombytes(seed, 32);
	
	fastrandombytes_setseed(seed);
	
	head = NORM_BATCH;

	for (i = 0; i < TOTAL; i++)
	{
		center[i] = (double)(rand()) / (double)(RAND_MAX);
	}
	
	cycle1 = cpucycles();
	for (i = 0; i < TOTAL; i++)
	{
		sample[i] = discrete_gaussian(center[i]);
	}
	cycle2 = cpucycles();
	
	for (i = 0; i < TOTAL; i++)
	{
		printf("%lld ", sample[i]);
	}
	printf("\n");
	printf("cycle:%lld\n", cycle2 - cycle1);
}
}
