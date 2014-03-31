/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2005       CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"

void bmgs_relax(const int relax_method, const bmgsstencil* s, double* a, double* b,
    const double* src, const double w)
{

if (relax_method == 1)
{
     /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
        a contains the temporary array holding also the boundary values. */

  // Coefficient needed multiple times later
  const double coef = 1.0/s->coefs[0];

  // The number of steps in each direction
  long nstep[3] = {s->n[0], s->n[1], s->n[2]};

  a += (s->j[0] + s->j[1] + s->j[2]) / 2;

  for (int i0 = 0; i0 < nstep[0]; i0++)
    {

      for (int i1 = 0; i1 < nstep[1]; i1++)
        {

          for (int i2 = 0; i2 < nstep[2]; i2++)
            {
              double x = 0.0;
              for (int c = 1; c < s->ncoefs; c++)
                x += a[s->offsets[c]] * s->coefs[c];
              x = (*src - x) * coef;
              *b++ = x;
              *a++ = x;
              src++;
            }
          a += s->j[2];
        }
      a += s->j[1];
    }

}
else
{
     /* Weighted Jacobi relaxation for the equation "operator" b = src
        a contains the temporariry array holding also the boundary values. */

  double temp;
  a += (s->j[0] + s->j[1] + s->j[2]) / 2;
  for (int i0 = 0; i0 < s->n[0]; i0++)
    {
      for (int i1 = 0; i1 < s->n[1]; i1++)
        {
          for (int i2 = 0; i2 < s->n[2]; i2++)
            {
              double x = 0.0;
              for (int c = 1; c < s->ncoefs; c++)
                x += a[s->offsets[c]] * s->coefs[c];
              temp = (1.0 - w) * *b + w * (*src - x)/s->coefs[0];
              *b++ = temp;
              a++;
              src++;
            }
          a += s->j[2];
        }
      a += s->j[1];
    }
}

}
