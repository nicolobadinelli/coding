/** @file harmonic.c Documented harmonic module
 *
 * Julien Lesgourgues, 1.11.2019
 *
 * This module computes the harmonic power spectra \f$ C_l^{X} \f$'s
 * given the transfer functions and the primordial spectra.
 *
 * The following functions can be called from other modules:
 *
 * -# harmonic_init() at the beginning (but after transfer_init())
 * -# harmonic_cl_at_l() at any time for computing individual \f$ C_l \f$'s at any l
 * -# harmonic_free() at the end
 */


#include "harmonic.h"
#include "parallel.h"

/* ============================================================ */
/* Visibility-weighted helpers for birefringence tomography      */
/* ============================================================ */

/**
 * Find the two local maxima of the visibility function g(tau) and the
 * minimum between them, using the same logic already used upstream in
 * perturbations.c for the alpha-source split.
 *
 * The first peak is interpreted as recombination, the second as
 * reionization. The minimum between them defines the split point.
 */
/* Internal workspaces for split reco/reio EE and TE block spectra */
static double ** harmonic_ee_reco_block = NULL;
static double ** harmonic_ee_reio_block = NULL;
static double ** harmonic_te_reco_block = NULL;
static double ** harmonic_te_reio_block = NULL;

static int harmonic_find_visibility_split(
                                          struct thermodynamics * pth,
                                          int * index_peak_reco,
                                          int * index_peak_reio,
                                          int * index_valley
                                          ) {

  int index_th;
  int first_peak = -1;
  int second_peak = -1;
  int valley = -1;
  double g_left;
  double g_mid;
  double g_right;
  double g_valley;

  for (index_th = 1; index_th < pth->tt_size - 1; index_th++) {
    g_left  = pth->thermodynamics_table[(index_th-1)*pth->th_size + pth->index_th_g];
    g_mid   = pth->thermodynamics_table[index_th*pth->th_size + pth->index_th_g];
    g_right = pth->thermodynamics_table[(index_th+1)*pth->th_size + pth->index_th_g];

    if ((g_mid >= g_left) && (g_mid >= g_right) && (g_mid > 0.0)) {
      if (first_peak < 0) {
        first_peak = index_th;
      }
      else {
        second_peak = index_th;
        break;
      }
    }
  }

  class_test((first_peak < 0) || (second_peak < 0),
             pth->error_message,
             "Could not identify recombination and reionization peaks in the visibility function for harmonic-space alpha weighting.");

  valley = first_peak;
  g_valley = pth->thermodynamics_table[first_peak*pth->th_size + pth->index_th_g];

  for (index_th = first_peak; index_th <= second_peak; index_th++) {
    g_mid = pth->thermodynamics_table[index_th*pth->th_size + pth->index_th_g];
    if (g_mid < g_valley) {
      g_valley = g_mid;
      valley = index_th;
    }
  }

  *index_peak_reco = first_peak;
  *index_peak_reio = second_peak;
  *index_valley = valley;

  return _SUCCESS_;
}

/**
 * Compute the visibility-weighted average of chi0 over either the
 * recombination or reionization part of the visibility function.
 *
 * split_flag = 0 -> recombination side (indices <= valley)
 * split_flag = 1 -> reionization side (indices >  valley)
 */
static int harmonic_visibility_weighted_chi0(
                                             struct background * pba,
                                             struct thermodynamics * pth,
                                             int split_flag,
                                             double * chi0_avg
                                             ) {

  int index_th;
  int index_peak_reco;
  int index_peak_reio;
  int index_valley;
  int index_start;
  int index_end;
  int last_index_back;
  double * pvecback;
  double tau;
  double tau_prev = 0.0;
  double z;
  double g;
  double g_prev = 0.0;
  double chi0;
  double chi0_prev = 0.0;
  double delta_tau;
  double weight_sum = 0.0;
  double weighted_chi_sum = 0.0;

  class_alloc(pvecback,
              pba->bg_size * sizeof(double),
              pth->error_message);

  class_call(harmonic_find_visibility_split(pth,
                                            &index_peak_reco,
                                            &index_peak_reio,
                                            &index_valley),
             pth->error_message,
             pth->error_message);

  index_start = (split_flag == 0) ? 0 : index_valley + 1;
  index_end   = (split_flag == 0) ? index_valley : pth->tt_size - 1;

  class_test(index_start >= index_end,
             pth->error_message,
             "Visibility-weighted chi0 average failed because the selected visibility window is too narrow for trapezoidal integration.");

  last_index_back = 0;

  for (index_th = index_start; index_th <= index_end; index_th++) {

    z = pth->z_table[index_th];
    g = pth->thermodynamics_table[index_th*pth->th_size + pth->index_th_g];

    class_call(background_tau_of_z(pba,
                                   z,
                                   &tau),
               pba->error_message,
               pth->error_message);

    class_call(background_at_tau(pba,
                                 tau,
                                 long_info,
                                 inter_normal,
                                 &last_index_back,
                                 pvecback),
               pba->error_message,
               pth->error_message);

    chi0 = pvecback[pba->index_bg_chi0];

    if (index_th == index_start) {
      tau_prev = tau;
      g_prev = g;
      chi0_prev = chi0;
      continue;
    }

    delta_tau = fabs(tau - tau_prev);

    weighted_chi_sum += 0.5 * (g_prev * chi0_prev + g * chi0) * delta_tau;
    weight_sum       += 0.5 * (g_prev + g) * delta_tau;

    tau_prev = tau;
    g_prev = g;
    chi0_prev = chi0;
  }

  free(pvecback);

  class_test(weight_sum <= 0.0,
             pth->error_message,
             "Visibility-weighted chi0 average failed because the selected visibility window has zero total weight.");

  *chi0_avg = weighted_chi_sum / weight_sum;

  return _SUCCESS_;
}

/**
 * Compute the isotropic birefringence angles alpha0_reco and alpha0_reio
 * with the visibility-weighted prescription discussed in the paper, i.e.
 * replacing chi0(tau_x) with the visibility-weighted average around each
 * peak rather than using a single sharp time.
 */
static int harmonic_compute_alpha0_visibility_weighted(
                                                       struct precision * ppr,
                                                       struct background * pba,
                                                       struct thermodynamics * pth,
                                                       double * alpha0_reco,
                                                       double * alpha0_reio
                                                       ) {

  double chi0_today;
  double chi0_reco_avg;
  double chi0_reio_avg;
  double * pvecback;
  int last_index_back;

  class_alloc(pvecback,
              pba->bg_size * sizeof(double),
              pth->error_message);

  last_index_back = 0;
  class_call(background_at_tau(pba,
                               pba->conformal_age,
                               long_info,
                               inter_normal,
                               &last_index_back,
                               pvecback),
             pba->error_message,
             pth->error_message);

  chi0_today = pvecback[pba->index_bg_chi0];

  free(pvecback);

  class_call(harmonic_visibility_weighted_chi0(pba,
                                               pth,
                                               0,
                                               &chi0_reco_avg),
             pth->error_message,
             pth->error_message);

  class_call(harmonic_visibility_weighted_chi0(pba,
                                               pth,
                                               1,
                                               &chi0_reio_avg),
             pth->error_message,
             pth->error_message);

  *alpha0_reco = 0.5 * ppr->lambda_over_f * (chi0_today - chi0_reco_avg);
  *alpha0_reio = 0.5 * ppr->lambda_over_f * (chi0_today - chi0_reio_avg);

  return _SUCCESS_;
}

/*
 * WARNING FOR FUTURE EDITS:
 *
 * The exact rotated spectra of the paper cannot be obtained by editing only
 * harmonic_apply_birefringence_rotation(). The paper-level result requires:
 *   - separate reco/reio EE and TE block spectra, not only the total EE/TE;
 *   - the anisotropic Wigner-3j sums of Appendix A;
 *   - a post-processing step after the ordinary C_l arrays have been built.
 *
 * Therefore this function should be treated only as a placeholder sanity
 * rotation until the full Wigner-based pipeline is implemented elsewhere.
 */

static int harmonic_apply_birefringence_rotation(
                                                        struct precision * ppr,
                                                        struct background * pba,
                                                        struct thermodynamics * pth,
                                                        struct perturbations * ppt,
                                                        struct harmonic * phr,
                                                        int index_md
                                                        ) {

  int index_l;
  int index_ic1;
  int index_ic2;
  int index_ic1_ic2;
  int index_L;
  int offset_l;
  int offset_L;
  double alpha0_reco;
  double alpha0_reio;
  double V_alpha_reco;
  double V_alpha_reio;
  double TE;
  double EE;
  double c2_reco;
  double s2_reco;
  double c2_reio;
  double s2_reio;
  double s4_reco;
  double s4_reio;
  double pref_te_reco;
  double pref_te_reio;
  double pref_pol_reco;
  double pref_pol_reio;
  int l_max_rot;

  /*
   * IMPORTANT:
   * This routine is only a simple isotropic/sanity rotation.
   * It is NOT the exact paper-level computation with Wigner-3j sums.
   *
   * In particular, the exact implementation in the paper requires
   * separate reco/reio EE and TE block spectra and the anisotropic
   * Wigner-coupling sums of Appendix A. Those ingredients are not
   * available inside this function alone.
   *
   * To avoid producing misleading spectra, keep this helper limited to
   * a conservative fallback: use it only when the rotated output slots
   * exist, and apply a purely isotropic rescaling to the total EE/TE.
   * This should be treated as a placeholder, not as the final result.
   */

  if ((index_md != phr->index_md_scalars) ||
      (ppt->has_source_alpha == _FALSE_) ||
      (ppt->has_scalars == _FALSE_)) {
    return _SUCCESS_;
  }

  class_call(harmonic_compute_alpha0_visibility_weighted(ppr,
                                                         pba,
                                                         pth,
                                                         &alpha0_reco,
                                                         &alpha0_reio),
             pth->error_message,
             phr->error_message);

  c2_reco = cos(2.0 * alpha0_reco);
  s2_reco = sin(2.0 * alpha0_reco);
  c2_reio = cos(2.0 * alpha0_reio);
  s2_reio = sin(2.0 * alpha0_reio);
  s4_reco = sin(4.0 * alpha0_reco);
  s4_reio = sin(4.0 * alpha0_reio);

  l_max_rot = phr->l_max_ct[index_md][phr->index_ct_ee_rot];
  if (phr->has_ee_rot == _FALSE_) {
    if (phr->has_te_rot == _TRUE_)
      l_max_rot = phr->l_max_ct[index_md][phr->index_ct_te_rot];
    else if (phr->has_tb_rot == _TRUE_)
      l_max_rot = phr->l_max_ct[index_md][phr->index_ct_tb_rot];
    else if (phr->has_bb_rot == _TRUE_)
      l_max_rot = phr->l_max_ct[index_md][phr->index_ct_bb_rot];
    else if (phr->has_eb_rot == _TRUE_)
      l_max_rot = phr->l_max_ct[index_md][phr->index_ct_eb_rot];
  }

  for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
    for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {

      index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);

      V_alpha_reco = 0.0;
      V_alpha_reio = 0.0;

      if (phr->has_aa_reco == _TRUE_) {
        for (index_L = 0; index_L < phr->l_size[index_md]; index_L++) {
          offset_L = (index_L * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size;
          V_alpha_reco += (2.0 * phr->l[index_L] + 1.0) / (4.0 * _PI_) *
            phr->cl[index_md][offset_L + phr->index_ct_aa_reco];
        }
      }

      if (phr->has_aa_reio == _TRUE_) {
        for (index_L = 0; index_L < phr->l_size[index_md]; index_L++) {
          offset_L = (index_L * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size;
          V_alpha_reio += (2.0 * phr->l[index_L] + 1.0) / (4.0 * _PI_) *
            phr->cl[index_md][offset_L + phr->index_ct_aa_reio];
        }
      }

      pref_te_reco = 1.0 - 2.0 * V_alpha_reco;
      pref_te_reio = 1.0 - 2.0 * V_alpha_reio;
      pref_pol_reco = 1.0 - 4.0 * V_alpha_reco;
      pref_pol_reio = 1.0 - 4.0 * V_alpha_reio;

      for (index_l = 0; index_l < phr->l_size[index_md]; index_l++) {

        if (phr->l[index_l] > (double)l_max_rot)
          break;

        offset_l = (index_l * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size;

        if ((harmonic_te_reco_block != NULL) &&
            (harmonic_te_reio_block != NULL) &&
            (harmonic_ee_reco_block != NULL) &&
            (harmonic_ee_reio_block != NULL) &&
            (harmonic_te_reco_block[index_md] != NULL) &&
            (harmonic_te_reio_block[index_md] != NULL) &&
            (harmonic_ee_reco_block[index_md] != NULL) &&
            (harmonic_ee_reio_block[index_md] != NULL)) {

          TE = harmonic_te_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
            + harmonic_te_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];

          EE = harmonic_ee_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
            + harmonic_ee_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
        }
        else {
          TE = (phr->has_te == _TRUE_) ? phr->cl[index_md][offset_l + phr->index_ct_te] : 0.0;
          EE = (phr->has_ee == _TRUE_) ? phr->cl[index_md][offset_l + phr->index_ct_ee] : 0.0;
        }
        /*
         * Fallback approximation:
         * since we do not have the separate reco/reio TE and EE blocks here,
         * do NOT treat the same total TE/EE as if it were both reco and reio.
         * Instead use a single effective isotropic rotation built from the sum
         * of the two epoch contributions at the level of the angles/variances.
         */
                if (phr->has_te_rot == _TRUE_) {
                  if ((harmonic_te_reco_block != NULL) &&
                        (harmonic_te_reio_block != NULL) &&
                        (harmonic_te_reco_block[index_md] != NULL) &&
                        (harmonic_te_reio_block[index_md] != NULL)) {
                      phr->cl[index_md][offset_l + phr->index_ct_te_rot] =
                        pref_te_reco * c2_reco *
                        harmonic_te_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
                        +
                        pref_te_reio * c2_reio *
                        harmonic_te_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
                    }
                    else {
                      phr->cl[index_md][offset_l + phr->index_ct_te_rot] =
                        0.5 * (pref_te_reco * c2_reco + pref_te_reio * c2_reio) * TE;
                    }
                }

                if (phr->has_tb_rot == _TRUE_) {
                  if ((harmonic_te_reco_block != NULL) &&
                        (harmonic_te_reio_block != NULL) &&
                        (harmonic_te_reco_block[index_md] != NULL) &&
                        (harmonic_te_reio_block[index_md] != NULL)) {
                      phr->cl[index_md][offset_l + phr->index_ct_tb_rot] =
                        pref_te_reco * s2_reco *
                        harmonic_te_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
                        +
                        pref_te_reio * s2_reio *
                        harmonic_te_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
                    }
                    else {
                      phr->cl[index_md][offset_l + phr->index_ct_tb_rot] =
                        0.5 * (pref_te_reco * s2_reco + pref_te_reio * s2_reio) * TE;
                    }
                }

               if (phr->has_ee_rot == _TRUE_) {
                if ((harmonic_ee_reco_block != NULL) &&
                    (harmonic_ee_reio_block != NULL) &&
                    (harmonic_ee_reco_block[index_md] != NULL) &&
                    (harmonic_ee_reio_block[index_md] != NULL)) {
                  phr->cl[index_md][offset_l + phr->index_ct_ee_rot] =
                    pref_pol_reco * c2_reco * c2_reco *
                    harmonic_ee_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
                    +
                    pref_pol_reio * c2_reio * c2_reio *
                    harmonic_ee_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
                }
                else {
                  phr->cl[index_md][offset_l + phr->index_ct_ee_rot] =
                    0.5 * (pref_pol_reco * c2_reco * c2_reco +
                          pref_pol_reio * c2_reio * c2_reio) * EE;
                }
              }

                if (phr->has_bb_rot == _TRUE_) {
                  if ((harmonic_ee_reco_block != NULL) &&
                      (harmonic_ee_reio_block != NULL) &&
                      (harmonic_ee_reco_block[index_md] != NULL) &&
                      (harmonic_ee_reio_block[index_md] != NULL)) {
                    phr->cl[index_md][offset_l + phr->index_ct_bb_rot] =
                      pref_pol_reco * s2_reco * s2_reco *
                      harmonic_ee_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
                      +
                      pref_pol_reio * s2_reio * s2_reio *
                      harmonic_ee_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
                  }
                  else {
                    phr->cl[index_md][offset_l + phr->index_ct_bb_rot] =
                      0.5 * (pref_pol_reco * s2_reco * s2_reco +
                            pref_pol_reio * s2_reio * s2_reio) * EE;
                  }
                }

                if (phr->has_eb_rot == _TRUE_) {
                  if ((harmonic_ee_reco_block != NULL) &&
                      (harmonic_ee_reio_block != NULL) &&
                      (harmonic_ee_reco_block[index_md] != NULL) &&
                      (harmonic_ee_reio_block[index_md] != NULL)) {
                    phr->cl[index_md][offset_l + phr->index_ct_eb_rot] =
                      0.5 * pref_pol_reco * s4_reco *
                      harmonic_ee_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2]
                      +
                      0.5 * pref_pol_reio * s4_reio *
                      harmonic_ee_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2];
                  }
                  else {
                    phr->cl[index_md][offset_l + phr->index_ct_eb_rot] =
                      0.25 * (pref_pol_reco * s4_reco +
                              pref_pol_reio * s4_reio) * EE;
                  }
                }
      }
    }
  }

  return _SUCCESS_;
}

/**
 * Anisotropy power spectra \f$ C_l\f$'s for all types, modes and initial conditions.
 * This routine evaluates all the \f$C_l\f$'s at a given value of l by
 * interpolating in the pre-computed table. When relevant, it also
 * sums over all initial conditions for each mode, and over all modes.
 *
 * This function can be
 * called from whatever module at whatever time, provided that
 * harmonic_init() has been called before, and harmonic_free() has not
 * been called yet.
 *
 * @param phr        Input: pointer to harmonic structure (containing pre-computed table)
 * @param l          Input: multipole number
 * @param cl_tot     Output: total \f$C_l\f$'s for all types (TT, TE, EE, etc..)
 * @param cl_md      Output: \f$C_l\f$'s for all types (TT, TE, EE, etc..) decomposed mode by mode (scalar, tensor, ...) when relevant
 * @param cl_md_ic   Output: \f$C_l\f$'s for all types (TT, TE, EE, etc..) decomposed by pairs of initial conditions (adiabatic, isocurvatures) for each mode (usually, only for the scalar mode) when relevant
 * @return the error status
 */

int harmonic_cl_at_l(
                     struct harmonic * phr,
                     double l,
                     double * cl_tot,    /* array with argument cl_tot[index_ct] (must be already allocated) */
                     double * * cl_md,   /* array with argument cl_md[index_md][index_ct] (must be already allocated only if several modes) */
                     double * * cl_md_ic /* array with argument cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct] (must be already allocated for a given mode only if several ic's) */
                     ) {

  /** Summary: */

  /** - define local variables */

  int last_index;
  int index_md;
  int index_ic1,index_ic2,index_ic1_ic2;
  int index_ct;

  /** - (a) treat case in which there is only one mode and one initial condition.
      Then, only cl_tot needs to be filled. */

  if ((phr->md_size == 1) && (phr->ic_size[0] == 1)) {
    index_md = 0;
    if ((int)l <= phr->l[phr->l_size[index_md]-1]) {

      /* interpolate at l */
      class_call(array_interpolate_spline(phr->l,
                                          phr->l_size[index_md],
                                          phr->cl[index_md],
                                          phr->ddcl[index_md],
                                          phr->ct_size,
                                          l,
                                          &last_index,
                                          cl_tot,
                                          phr->ct_size,
                                          phr->error_message),
                 phr->error_message,
                 phr->error_message);

      /* set to zero for the types such that l<l_max */
      for (index_ct=0; index_ct<phr->ct_size; index_ct++)
        if ((int)l > phr->l_max_ct[index_md][index_ct])
          cl_tot[index_ct]=0.;
    }
    else {
      for (index_ct=0; index_ct<phr->ct_size; index_ct++)
        cl_tot[index_ct]=0.;
    }
  }

  /** - (b) treat case in which there is only one mode
      with several initial condition.
      Fill cl_md_ic[index_md=0] and sum it to get cl_tot. */

  if ((phr->md_size == 1) && (phr->ic_size[0] > 1)) {
    index_md = 0;
    for (index_ct=0; index_ct<phr->ct_size; index_ct++)
      cl_tot[index_ct]=0.;
    for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
      for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {
        index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);
        if (((int)l <= phr->l[phr->l_size[index_md]-1]) &&
            (phr->is_non_zero[index_md][index_ic1_ic2] == _TRUE_)) {

          class_call(array_interpolate_spline(phr->l,
                                              phr->l_size[index_md],
                                              phr->cl[index_md],
                                              phr->ddcl[index_md],
                                              phr->ic_ic_size[index_md]*phr->ct_size,
                                              l,
                                              &last_index,
                                              cl_md_ic[index_md],
                                              phr->ic_ic_size[index_md]*phr->ct_size,
                                              phr->error_message),
                     phr->error_message,
                     phr->error_message);

          for (index_ct=0; index_ct<phr->ct_size; index_ct++)
            if ((int)l > phr->l_max_ct[index_md][index_ct])
              cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct]=0.;
        }
        else {
          for (index_ct=0; index_ct<phr->ct_size; index_ct++)
            cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct]=0.;
        }

        /* compute cl_tot by summing over cl_md_ic */
        for (index_ct=0; index_ct<phr->ct_size; index_ct++) {
          if (index_ic1 == index_ic2)
            cl_tot[index_ct]+=cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct];
          else
            cl_tot[index_ct]+=2.*cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct];
        }
      }
    }
  }

  /** - (c) loop over modes */

  if (phr->md_size > 1) {

    for (index_ct=0; index_ct<phr->ct_size; index_ct++)
      cl_tot[index_ct]=0.;

    for (index_md = 0; index_md < phr->md_size; index_md++) {

      /** - --> (c.1.) treat case in which the mode under consideration
          has only one initial condition.
          Fill cl_md[index_md]. */

      if (phr->ic_size[index_md] == 1) {
        if ((int)l <= phr->l[phr->l_size[index_md]-1]) {

          class_call(array_interpolate_spline(phr->l,
                                              phr->l_size[index_md],
                                              phr->cl[index_md],
                                              phr->ddcl[index_md],
                                              phr->ct_size,
                                              l,
                                              &last_index,
                                              cl_md[index_md],
                                              phr->ct_size,
                                              phr->error_message),
                     phr->error_message,
                     phr->error_message);

          for (index_ct=0; index_ct<phr->ct_size; index_ct++)
            if ((int)l > phr->l_max_ct[index_md][index_ct])
              cl_md[index_md][index_ct]=0.;
        }
        else {
          for (index_ct=0; index_ct<phr->ct_size; index_ct++)
            cl_md[index_md][index_ct]=0.;
        }
      }

      /** - --> (c.2.) treat case in which the mode under consideration
          has several initial conditions.
          Fill cl_md_ic[index_md] and sum it to get cl_md[index_md] */

      if (phr->ic_size[index_md] > 1) {

        if ((int)l <= phr->l[phr->l_size[index_md]-1]) {

          /* interpolate all ic and ct */
          class_call(array_interpolate_spline(phr->l,
                                              phr->l_size[index_md],
                                              phr->cl[index_md],
                                              phr->ddcl[index_md],
                                              phr->ic_ic_size[index_md]*phr->ct_size,
                                              l,
                                              &last_index,
                                              cl_md_ic[index_md],
                                              phr->ic_ic_size[index_md]*phr->ct_size,
                                              phr->error_message),
                     phr->error_message,
                     phr->error_message);

          /* set to zero some of the components */
          for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
            for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {
              index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);
              for (index_ct=0; index_ct<phr->ct_size; index_ct++) {

                if (((int)l > phr->l_max_ct[index_md][index_ct]) || (phr->is_non_zero[index_md][index_ic1_ic2] == _FALSE_))
                  cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct]=0.;
              }
            }
          }
        }
        /* if l was too big, set anyway all components to zero */
        else {
          for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
            for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {
              index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);
              for (index_ct=0; index_ct<phr->ct_size; index_ct++) {
                cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct]=0.;
              }
            }
          }
        }

        /* sum up all ic for each mode */

        for (index_ct=0; index_ct<phr->ct_size; index_ct++) {

          cl_md[index_md][index_ct]=0.;

          for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
            for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {
              index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);

              if (index_ic1 == index_ic2)
                cl_md[index_md][index_ct]+=cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct];
              else
                cl_md[index_md][index_ct]+=2.*cl_md_ic[index_md][index_ic1_ic2*phr->ct_size+index_ct];
            }
          }
        }
      }

      /** - --> (c.3.) add contribution of cl_md[index_md] to cl_tot */

      for (index_ct=0; index_ct<phr->ct_size; index_ct++)
        cl_tot[index_ct]+=cl_md[index_md][index_ct];
    }
  }

  return _SUCCESS_;

}

/**
 * This routine initializes the harmonic structure (in particular,
 * computes table of anisotropy and Fourier spectra \f$ C_l^{X}, P(k), ... \f$)
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure (will provide H, Omega_m at redshift of interest)
 * @param ppt Input: pointer to perturbation structure
 * @param ptr Input: pointer to transfer structure
 * @param ppm Input: pointer to primordial structure
 * @param pfo Input: pointer to fourier structure
 * @param phr Output: pointer to initialized harmonic structure
 * @return the error status
 */

int harmonic_init(
                  struct precision * ppr,
                  struct background * pba,
                  struct thermodynamics * pth,
                  struct perturbations * ppt,
                  struct primordial * ppm,
                  struct fourier * pfo,
                  struct transfer * ptr,
                  struct harmonic * phr
                  ) {

  /** Summary: */

  /** - check that we really want to compute at least one spectrum */

  if (ppt->has_cls == _FALSE_) {
    phr->md_size = 0;
    if (phr->harmonic_verbose > 0)
      printf("No spectra requested. Spectra module skipped.\n");
    return _SUCCESS_;
  }
  else {
    if (phr->harmonic_verbose > 0)
      printf("Computing unlensed harmonic spectra\n");
  }

  /** - initialize indices and allocate some of the arrays in the
      harmonic structure */

  class_call(harmonic_indices(pba,ppt,ptr,ppm,phr),
             phr->error_message,
             phr->error_message);

  /** - deal with \f$ C_l\f$'s, if any */

  if (ppt->has_cls == _TRUE_) {

    class_call(harmonic_cls(ppr,pba,pth,ppt,ptr,ppm,phr),
               phr->error_message,
               phr->error_message);

  }
  else {
    phr->ct_size=0;
  }

  /** - a pointer to the fourier structure is stored in the spectra
      structure. This odd, unusual and unelegant feature has been
      introduced in v2.8 in order to keep in use some deprecated
      functions harmonic_pk_...() that are now pointing at new
      function fourier_pk_...(). In the future, if the deprecated
      functions are removed, it will be possible to remove also this
      pointer. */

  phr->pfo = pfo;
  phr->is_allocated = _TRUE_;

  return _SUCCESS_;
}

/**
 * This routine frees all the memory space allocated by harmonic_init().
 *
 * To be called at the end of each run, only when no further calls to
 * harmonic_cls_at_l(), harmonic_pk_at_z(), harmonic_pk_at_k_and_z() are needed.
 *
 * @param phr Input: pointer to harmonic structure (which fields must be freed)
 * @return the error status
 */

int harmonic_free(
                  struct harmonic * phr
                  ) {

  int index_md;

  if (phr->md_size > 0) {
    if (phr->ct_size > 0) {

      for (index_md = 0; index_md < phr->md_size; index_md++) {
        free(phr->l_max_ct[index_md]);
        free(phr->cl[index_md]);
        free(phr->ddcl[index_md]);
      }
      free(phr->l);
      free(phr->l_size);
      free(phr->l_max_ct);
      free(phr->l_max);
      free(phr->cl);
      free(phr->ddcl);
    }

    for (index_md=0; index_md < phr->md_size; index_md++)
      free(phr->is_non_zero[index_md]);

    free(phr->is_non_zero);
    free(phr->ic_size);
    free(phr->ic_ic_size);

  
    if (harmonic_ee_reco_block != NULL) {
    free(harmonic_ee_reco_block);
    harmonic_ee_reco_block = NULL;
  }
  if (harmonic_ee_reio_block != NULL) {
    free(harmonic_ee_reio_block);
    harmonic_ee_reio_block = NULL;
  }
  if (harmonic_te_reco_block != NULL) {
    free(harmonic_te_reco_block);
    harmonic_te_reco_block = NULL;
  }
  if (harmonic_te_reio_block != NULL) {
    free(harmonic_te_reio_block);
    harmonic_te_reio_block = NULL;
  }

  phr->is_allocated = _FALSE_;
}
  return _SUCCESS_;

}

/**
 * This routine defines indices and allocates tables in the harmonic structure
 *
 * @param pba  Input: pointer to background structure
 * @param ppt  Input: pointer to perturbation structure
 * @param ptr  Input: pointer to transfer structure
 * @param ppm  Input: pointer to primordial structure
 * @param phr  Input/output: pointer to harmonic structure
 * @return the error status
 */

int harmonic_indices(
                     struct background * pba,
                     struct perturbations * ppt,
                     struct transfer * ptr,
                     struct primordial * ppm,
                     struct harmonic * phr
                     ){

  int index_ct;
  int index_md;
  int index_ic1_ic2;

  phr->md_size = ppt->md_size;
  if (ppt->has_scalars == _TRUE_)
    phr->index_md_scalars = ppt->index_md_scalars;

  class_alloc(phr->ic_size,
              sizeof(int)*phr->md_size,
              phr->error_message);

  class_alloc(phr->ic_ic_size,
              sizeof(int)*phr->md_size,
              phr->error_message);

  class_alloc(phr->is_non_zero,
              sizeof(short *)*phr->md_size,
              phr->error_message);

  for (index_md=0; index_md < phr->md_size; index_md++) {
    phr->ic_size[index_md] = ppm->ic_size[index_md];
    phr->ic_ic_size[index_md] = ppm->ic_ic_size[index_md];
    class_alloc(phr->is_non_zero[index_md],
                sizeof(short)*phr->ic_ic_size[index_md],
                phr->error_message);
    for (index_ic1_ic2=0; index_ic1_ic2 < phr->ic_ic_size[index_md]; index_ic1_ic2++)
      phr->is_non_zero[index_md][index_ic1_ic2] = ppm->is_non_zero[index_md][index_ic1_ic2];
  }

  if (ppt->has_cls == _TRUE_) {

    /* types of C_l's relevant for both scalars and tensors: TT, EE, TE */

    index_ct=0;

    if (ppt->has_cl_cmb_temperature == _TRUE_) {
      phr->has_tt = _TRUE_;
      phr->index_ct_tt=index_ct;
      index_ct++;
    }
    else {
      phr->has_tt = _FALSE_;
    }

    if (ppt->has_cl_cmb_polarization == _TRUE_) {
      phr->has_ee = _TRUE_;
      phr->index_ct_ee=index_ct;
      index_ct++;
    }
    else {
      phr->has_ee = _FALSE_;
    }

    if ((ppt->has_cl_cmb_temperature == _TRUE_) &&
        (ppt->has_cl_cmb_polarization == _TRUE_)) {
      phr->has_te = _TRUE_;
      phr->index_ct_te=index_ct;
      index_ct++;
    }
    else {
      phr->has_te = _FALSE_;
    }

    if (ppt->has_cl_cmb_polarization == _TRUE_) {
      phr->has_bb = _TRUE_;
      phr->index_ct_bb=index_ct;
      index_ct++;
    }
    else {
      phr->has_bb = _FALSE_;
    }

    /* types of C_l's relevant only for scalars: phi-phi, T-phi, E-phi, d-d, T-d */

    if ((ppt->has_cl_cmb_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_pp = _TRUE_;
      phr->index_ct_pp=index_ct;
      index_ct++;
    }
    else {
      phr->has_pp = _FALSE_;
    }

    if ((ppt->has_cl_cmb_temperature == _TRUE_) && (ppt->has_cl_cmb_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_tp = _TRUE_;
      phr->index_ct_tp=index_ct;
      index_ct++;
    }
    else {
      phr->has_tp = _FALSE_;
    }


    if ((ppt->has_cl_cmb_polarization == _TRUE_) && (ppt->has_cl_cmb_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_ep = _TRUE_;
      phr->index_ct_ep=index_ct;
      index_ct++;
    }
    else {
      phr->has_ep = _FALSE_;
    }

    /* alpha-alpha spectra */
    if ((ppt->has_source_alpha == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_aa = _TRUE_;
      phr->index_ct_aa = index_ct;
      index_ct++;

      phr->has_aa_reco = _TRUE_;
      phr->index_ct_aa_reco = index_ct;
      index_ct++;

      phr->has_aa_reio = _TRUE_;
      phr->index_ct_aa_reio = index_ct;
      index_ct++;
    }
    else {
      phr->has_aa = _FALSE_;
      phr->has_aa_reco = _FALSE_;
      phr->has_aa_reio = _FALSE_;
    }

    if ((ppt->has_source_alpha == _TRUE_) && (ppt->has_scalars == _TRUE_) && (ppt->has_cl_cmb_temperature == _TRUE_)) {
      phr->has_at = _TRUE_;
      phr->index_ct_at = index_ct;
      index_ct++;

      phr->has_at_reco = _TRUE_;
      phr->index_ct_at_reco = index_ct;
      index_ct++;

      phr->has_at_reio = _TRUE_;
      phr->index_ct_at_reio = index_ct;
      index_ct++;
    }
    else {
      phr->has_at = _FALSE_;
      phr->has_at_reco = _FALSE_;
      phr->has_at_reio = _FALSE_;
    }

    /* alpha-E spectra */
    if ((ppt->has_source_alpha == _TRUE_) && (ppt->has_scalars == _TRUE_) && (ppt->has_cl_cmb_polarization == _TRUE_)) {
      phr->has_ae = _TRUE_;
      phr->index_ct_ae = index_ct;
      index_ct++;

      phr->has_ae_reco = _TRUE_;
      phr->index_ct_ae_reco = index_ct;
      index_ct++;

      phr->has_ae_reio = _TRUE_;
      phr->index_ct_ae_reio = index_ct;
      index_ct++;
    }
    else {
      phr->has_ae = _FALSE_;
      phr->has_ae_reco = _FALSE_;
      phr->has_ae_reio = _FALSE_;
    }
    /* rotated CMB spectra from birefringence
       -----------------------------------------------------------------
       IMPORTANT: this block only declares output slots and their indices.
       The exact birefringence rotation with Wigner-3j symbols does NOT
       belong here. It must be applied later in harmonic.c, after the
       ordinary unlensed spectra have already been computed.

       In particular:
       - TE_rot and TB_rot require temperature + polarization spectra
       - EE_rot, BB_rot and EB_rot require polarization spectra
       - all rotated spectra also require the alpha source and scalar mode
    */
    if ((ppt->has_source_alpha == _TRUE_) &&
        (ppt->has_scalars == _TRUE_) &&
        (ppt->has_cl_cmb_temperature == _TRUE_) &&
        (ppt->has_cl_cmb_polarization == _TRUE_)) {

      phr->has_te_rot = _TRUE_;
      phr->index_ct_te_rot = index_ct;
      index_ct++;

      phr->has_tb_rot = _TRUE_;
      phr->index_ct_tb_rot = index_ct;
      index_ct++;
    }
    else {
      phr->has_te_rot = _FALSE_;
      phr->has_tb_rot = _FALSE_;
    }

    if ((ppt->has_source_alpha == _TRUE_) &&
        (ppt->has_scalars == _TRUE_) &&
        (ppt->has_cl_cmb_polarization == _TRUE_)) {

      phr->has_ee_rot = _TRUE_;
      phr->index_ct_ee_rot = index_ct;
      index_ct++;

      phr->has_bb_rot = _TRUE_;
      phr->index_ct_bb_rot = index_ct;
      index_ct++;

      phr->has_eb_rot = _TRUE_;
      phr->index_ct_eb_rot = index_ct;
      index_ct++;
    }
    else {
      phr->has_ee_rot = _FALSE_;
      phr->has_bb_rot = _FALSE_;
      phr->has_eb_rot = _FALSE_;
    }

    /* the computation of C_l^Td would require a very good sampling of
       transfer functions over a wide range, and a huge computation
       time. In the current version, we prefer to switch it off, rather
       than either slowing down the code considerably, or producing
       very inaccurate spectra.

       if ((ppt->has_cl_cmb_temperature == _TRUE_) && (ppt->has_cl_number_count == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
       phr->has_td = _TRUE_;
       phr->index_ct_td=index_ct;
       index_ct+=phr->d_size;
       }
       else {
       phr->has_td = _FALSE_;
       }
    */
    phr->has_td = _FALSE_;

    if ((ppt->has_cl_cmb_lensing_potential == _TRUE_) && (ppt->has_cl_number_count == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_pd = _TRUE_;
      phr->index_ct_pd=index_ct;
      index_ct+=phr->d_size;
    }
    else {
      phr->has_pd = _FALSE_;
    }

    if ((ppt->has_cl_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_ll = _TRUE_;
      phr->index_ct_ll=index_ct;
      index_ct+=(phr->d_size*(phr->d_size+1)-(phr->d_size-phr->non_diag)*(phr->d_size-1-phr->non_diag))/2;
    }
    else {
      phr->has_ll = _FALSE_;
    }

    /* the computation of C_l^Tl would require a very good sampling of
       transfer functions over a wide range, and a huge computation
       time. In the current version, we prefer to switch it off, rather
       than either slowing down the code considerably, or producing
       very inaccurate spectra.

       if ((ppt->has_cl_cmb_temperature == _TRUE_) && (ppt->has_cl_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
       phr->has_tl = _TRUE_;
       phr->index_ct_tl=index_ct;
       index_ct+=phr->d_size;
       }
       else {
       phr->has_tl = _FALSE_;
       }
    */
    phr->has_tl = _FALSE_;

    if ((ppt->has_cl_number_count == _TRUE_) && (ppt->has_cl_lensing_potential == _TRUE_) && (ppt->has_scalars == _TRUE_)) {
      phr->has_dl = _TRUE_;
      phr->index_ct_dl=index_ct;
      index_ct += phr->d_size*phr->d_size - (phr->d_size-phr->non_diag)*(phr->d_size-1-phr->non_diag);
    }
    else {
      phr->has_dl = _FALSE_;
    }

    phr->ct_size = index_ct;

    /* infer from input quantities the l_max for each mode and type,
       l_max_ct[index_md][index_type].  Maximize it over index_ct, and
       then over index_md. */

    class_alloc(phr->l_max,sizeof(int*)*phr->md_size,phr->error_message);
    class_alloc(phr->l_max_ct,sizeof(int*)*phr->md_size,phr->error_message);
    for (index_md=0; index_md<phr->md_size; index_md++) {
      class_calloc(phr->l_max_ct[index_md],phr->ct_size,sizeof(int),phr->error_message);
    }

    if (ppt->has_scalars == _TRUE_) {

      /* spectra computed up to l_scalar_max */

      if (phr->has_tt == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_tt] = ppt->l_scalar_max;
      if (phr->has_ee == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ee] = ppt->l_scalar_max;
      if (phr->has_te == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_te] = ppt->l_scalar_max;
      if (phr->has_pp == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_pp] = ppt->l_scalar_max;
      if (phr->has_tp == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_tp] = ppt->l_scalar_max;
      if (phr->has_ep == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ep] = ppt->l_scalar_max;
      if (phr->has_aa == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_aa] = ppt->l_scalar_max;
      if (phr->has_aa_reco == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_aa_reco] = ppt->l_scalar_max;
      if (phr->has_aa_reio == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_aa_reio] = ppt->l_scalar_max;
      if (phr->has_at == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_at] = ppt->l_scalar_max;
      if (phr->has_at_reco == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_at_reco] = ppt->l_scalar_max;
      if (phr->has_at_reio == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_at_reio] = ppt->l_scalar_max;

      if (phr->has_ae == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ae] = ppt->l_scalar_max;
      if (phr->has_ae_reco == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ae_reco] = ppt->l_scalar_max;
      if (phr->has_ae_reio == _TRUE_) phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ae_reio] = ppt->l_scalar_max;
      {
        int l_max_rot = 250;

        if (phr->has_te_rot == _TRUE_)
          phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_te_rot] = l_max_rot;

        if (phr->has_tb_rot == _TRUE_)
          phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_tb_rot] = l_max_rot;

        if (phr->has_ee_rot == _TRUE_)
          phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_ee_rot] = l_max_rot;

        if (phr->has_bb_rot == _TRUE_)
          phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_bb_rot] = l_max_rot;

        if (phr->has_eb_rot == _TRUE_)
          phr->l_max_ct[ppt->index_md_scalars][phr->index_ct_eb_rot] = l_max_rot;
      }

      /* spectra computed up to l_lss_max */

      if (phr->has_dd == _TRUE_)
        for (index_ct=phr->index_ct_dd;
             index_ct<phr->index_ct_dd+(phr->d_size*(phr->d_size+1)-(phr->d_size-phr->non_diag)*(phr->d_size-1-phr->non_diag))/2;
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = ppt->l_lss_max;

      if (phr->has_td == _TRUE_)
        for (index_ct=phr->index_ct_td;
             index_ct<phr->index_ct_td+phr->d_size;
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = MIN(ppt->l_scalar_max,ppt->l_lss_max);

      if (phr->has_pd == _TRUE_)
        for (index_ct=phr->index_ct_pd;
             index_ct<phr->index_ct_pd+phr->d_size;
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = MIN(ppt->l_scalar_max,ppt->l_lss_max);

      if (phr->has_ll == _TRUE_)
        for (index_ct=phr->index_ct_ll;
             index_ct<phr->index_ct_ll+(phr->d_size*(phr->d_size+1)-(phr->d_size-phr->non_diag)*(phr->d_size-1-phr->non_diag))/2;
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = ppt->l_lss_max;

      if (phr->has_tl == _TRUE_)
        for (index_ct=phr->index_ct_tl;
             index_ct<phr->index_ct_tl+phr->d_size;
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = MIN(ppt->l_scalar_max,ppt->l_lss_max);

      if (phr->has_dl == _TRUE_)
        for (index_ct=phr->index_ct_dl;
             index_ct < phr->index_ct_dl+(phr->d_size*phr->d_size - (phr->d_size-phr->non_diag)*(phr->d_size-1-phr->non_diag));
             index_ct++)
          phr->l_max_ct[ppt->index_md_scalars][index_ct] = ppt->l_lss_max;

    }
    if (ppt->has_tensors == _TRUE_) {

      /* spectra computed up to l_tensor_max */

      if (phr->has_tt == _TRUE_) phr->l_max_ct[ppt->index_md_tensors][phr->index_ct_tt] = ppt->l_tensor_max;
      if (phr->has_ee == _TRUE_) phr->l_max_ct[ppt->index_md_tensors][phr->index_ct_ee] = ppt->l_tensor_max;
      if (phr->has_te == _TRUE_) phr->l_max_ct[ppt->index_md_tensors][phr->index_ct_te] = ppt->l_tensor_max;
      if (phr->has_bb == _TRUE_) phr->l_max_ct[ppt->index_md_tensors][phr->index_ct_bb] = ppt->l_tensor_max;
    }

    /* maximizations */
    phr->l_max_tot = 0.;
    for (index_md=0; index_md < phr->md_size; index_md++) {
      phr->l_max[index_md] = 0.;
      for (index_ct=0.; index_ct<phr->ct_size; index_ct++)
        phr->l_max[index_md] = MAX(phr->l_max[index_md],phr->l_max_ct[index_md][index_ct]);
      phr->l_max_tot = MAX(phr->l_max_tot,phr->l_max[index_md]);
    }
  }

  return _SUCCESS_;

}

/**
 * This routine computes a table of values for all harmonic spectra \f$ C_l \f$'s,
 * given the transfer functions and primordial spectra.
 *
 * @param ppr Input: pointer to precision structure
 * @param pba Input: pointer to background structure
 * @param ppt Input: pointer to perturbation structure
 * @param ptr Input: pointer to transfer structure
 * @param ppm Input: pointer to primordial structure
 * @param phr Input/Output: pointer to harmonic structure
 * @return the error status
 */

int harmonic_cls(
                 struct precision * ppr,
                 struct background * pba,
                 struct thermodynamics * pth,
                 struct perturbations * ppt,
                 struct transfer * ptr,
                 struct primordial * ppm,
                 struct harmonic * phr
                 ) {

  /** Summary: */

  /** - define local variables */

  int index_md;
  int index_ic1,index_ic2,index_ic1_ic2;
  int index_l;
  int index_ct;
  int index_tmp;
  int cl_integrand_num_columns;

  /** - allocate pointers to arrays where results will be stored */

  class_alloc(phr->l_size,sizeof(int)*phr->md_size,phr->error_message);
  class_alloc(phr->cl,sizeof(double *)*phr->md_size,phr->error_message);
  class_alloc(phr->ddcl,sizeof(double *)*phr->md_size,phr->error_message);
  class_alloc(harmonic_ee_reco_block,sizeof(double *)*phr->md_size,phr->error_message);
  class_alloc(harmonic_ee_reio_block,sizeof(double *)*phr->md_size,phr->error_message);
  class_alloc(harmonic_te_reco_block,sizeof(double *)*phr->md_size,phr->error_message);
  class_alloc(harmonic_te_reio_block,sizeof(double *)*phr->md_size,phr->error_message);

  phr->l_size_max = ptr->l_size_max;
  class_alloc(phr->l,sizeof(double)*phr->l_size_max,phr->error_message);

  /** - store values of l */
  for (index_l=0; index_l < phr->l_size_max; index_l++) {
    phr->l[index_l] = (double)ptr->l[index_l];
  }

  /** - loop over modes (scalar, tensors, etc). For each mode: */

  for (index_md = 0; index_md < phr->md_size; index_md++) {

    /** - --> (a) store number of l values for this mode */

    phr->l_size[index_md] = ptr->l_size[index_md];

    /** - --> (b) allocate arrays where results will be stored */

    class_alloc(phr->cl[index_md],sizeof(double)*phr->l_size[index_md]*phr->ct_size*phr->ic_ic_size[index_md],phr->error_message);
    class_alloc(phr->ddcl[index_md],sizeof(double)*phr->l_size[index_md]*phr->ct_size*phr->ic_ic_size[index_md],phr->error_message);
    class_alloc(harmonic_ee_reco_block[index_md],
            sizeof(double)*phr->l_size[index_md]*phr->ic_ic_size[index_md],
            phr->error_message);

    class_alloc(harmonic_ee_reio_block[index_md],
                sizeof(double)*phr->l_size[index_md]*phr->ic_ic_size[index_md],
                phr->error_message);

    class_alloc(harmonic_te_reco_block[index_md],
                sizeof(double)*phr->l_size[index_md]*phr->ic_ic_size[index_md],
                phr->error_message);

    class_alloc(harmonic_te_reio_block[index_md],
                sizeof(double)*phr->l_size[index_md]*phr->ic_ic_size[index_md],
                phr->error_message);
        cl_integrand_num_columns = 1+phr->ct_size*2; /* one for k, ct_size for each type, ct_size for each second derivative of each type */
    for (index_tmp = 0; index_tmp < phr->l_size[index_md]*phr->ic_ic_size[index_md]; index_tmp++) {
      harmonic_ee_reco_block[index_md][index_tmp] = 0.0;
      harmonic_ee_reio_block[index_md][index_tmp] = 0.0;
      harmonic_te_reco_block[index_md][index_tmp] = 0.0;
      harmonic_te_reio_block[index_md][index_tmp] = 0.0;
    }
    /** - --> (c) loop over initial conditions */

    class_setup_parallel();

    for (index_ic1 = 0; index_ic1 < phr->ic_size[index_md]; index_ic1++) {
      for (index_ic2 = index_ic1; index_ic2 < phr->ic_size[index_md]; index_ic2++) {
        index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);

        /* non-diagonal coefficients should be computed only if non-zero correlation */
        if (phr->is_non_zero[index_md][index_ic1_ic2] == _TRUE_) {

          /** - ---> loop over l values defined in the transfer module.
              For each l, compute the \f$ C_l\f$'s for all types (TT, TE, ...)
              by convolving primordial spectra with transfer  functions.
              This elementary task is assigned to harmonic_compute_cl() */

          for (index_l=0; index_l < ptr->l_size[index_md]; index_l++) {

            class_run_parallel(=,

              double * cl_integrand; /* array with argument cl_integrand[index_k*cl_integrand_num_columns+1+phr->index_ct] */
              double * cl_integrand_limber; /* similar array with same columns but different number of lines (less k values) */
              double * transfer_ic1; /* array with argument transfer_ic1[index_tt] */
              double * transfer_ic2; /* idem */
              double * primordial_pk;  /* array with argument primordial_pk[index_ic_ic]*/


              class_alloc(cl_integrand,
                          ptr->q_size*cl_integrand_num_columns*sizeof(double),
                          phr->error_message);

              cl_integrand_limber = NULL;
              if (ptr->do_lcmb_full_limber == _TRUE_) {
                class_alloc(cl_integrand_limber,
                            ptr->q_size_limber*cl_integrand_num_columns*sizeof(double),
                            phr->error_message);
              }

              class_alloc(primordial_pk,
                          phr->ic_ic_size[index_md]*sizeof(double),
                          phr->error_message);

              class_alloc(transfer_ic1,
                          ptr->tt_size[index_md]*sizeof(double),
                          phr->error_message);

              class_alloc(transfer_ic2,
                          ptr->tt_size[index_md]*sizeof(double),
                          phr->error_message);

              class_call(harmonic_compute_cl(ppr,
                                             pba,
                                             ppt,
                                             ptr,
                                             ppm,
                                             phr,
                                             index_md,
                                             index_ic1,
                                             index_ic2,
                                             index_l,
                                             cl_integrand_num_columns,
                                             cl_integrand,
                                             cl_integrand_limber,
                                             primordial_pk,
                                             transfer_ic1,
                                             transfer_ic2),
                         phr->error_message,
                         phr->error_message);

              free(cl_integrand);
              if (ptr->do_lcmb_full_limber == _TRUE_) {
                free(cl_integrand_limber);
              }
              free(primordial_pk);
              free(transfer_ic1);
              free(transfer_ic2);

              return _SUCCESS_;
            );
          } /* end of loop over l */

        }
        else {

          /* set non-diagonal coefficients to zero if pair of ic's uncorrelated */

          for (index_l=0; index_l < ptr->l_size[index_md]; index_l++) {
            for (index_ct=0; index_ct<phr->ct_size; index_ct++) {
              phr->cl[index_md]
                [(index_l * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size + index_ct]
                = 0.;
            }
          }
        }
      }
    }

    class_finish_parallel();

    /*
     * NOTE:
     * The exact paper-level birefringence rotation using Wigner-3j sums
     * must be applied here, after phr->cl[index_md] has been filled with
     * the ordinary unlensed spectra and before the spline table is built.
     *
     * The current helper called below is only the post-processing stage.
     * The index-allocation logic in harmonic_indices() is not the place
     * where Wigner symbols should be implemented.
     */
    class_call(harmonic_apply_birefringence_rotation(ppr,
                                                            pba,
                                                            pth,
                                                            ppt,
                                                            phr,
                                                            index_md),
               phr->error_message,
               phr->error_message);

    /** - --> (d) now that for a given mode, all possible \f$ C_l\f$'s have been computed,
        compute second derivative of the array in which they are stored,
        in view of spline interpolation. */

    class_call(array_spline_table_lines(phr->l,
                                        phr->l_size[index_md],
                                        phr->cl[index_md],
                                        phr->ic_ic_size[index_md]*phr->ct_size,
                                        phr->ddcl[index_md],
                                        _SPLINE_EST_DERIV_,
                                        phr->error_message),
               phr->error_message,
               phr->error_message);
  }
    free(harmonic_ee_reco_block[index_md]);
    free(harmonic_ee_reio_block[index_md]);
    free(harmonic_te_reco_block[index_md]);
    free(harmonic_te_reio_block[index_md]);
    harmonic_ee_reco_block[index_md] = NULL;
    harmonic_ee_reio_block[index_md] = NULL;
    harmonic_te_reco_block[index_md] = NULL;
    harmonic_te_reio_block[index_md] = NULL;

  return _SUCCESS_;

}

/**
 * This routine computes the \f$ C_l\f$'s for a given mode, pair of initial conditions
 * and multipole, but for all types (TT, TE...), by convolving the
 * transfer functions with the primordial spectra.
 *
 * @param ppr           Input: pointer to precision structure
 * @param pba           Input: pointer to background structure
 * @param ppt           Input: pointer to perturbation structure
 * @param ptr           Input: pointer to transfer structure
 * @param ppm           Input: pointer to primordial structure
 * @param phr           Input/Output: pointer to harmonic structure (result stored here)
 * @param index_md      Input: index of mode under consideration
 * @param index_ic1     Input: index of first initial condition in the correlator
 * @param index_ic2     Input: index of second initial condition in the correlator
 * @param index_l       Input: index of multipole under consideration
 * @param cl_integrand_num_columns Input: number of columns in cl_integrand
 * @param cl_integrand  Input: an allocated workspace
 * @param cl_integrand_limber  Input: an allocated workspace for full Limber calculation
 * @param primordial_pk Input: table of primordial spectrum values
 * @param transfer_ic1  Input: table of transfer function values for first initial condition
 * @param transfer_ic2  Input: table of transfer function values for second initial condition
 * @return the error status
 */

int harmonic_compute_cl(
                        struct precision * ppr,
                        struct background * pba,
                        struct perturbations * ppt,
                        struct transfer * ptr,
                        struct primordial * ppm,
                        struct harmonic * phr,
                        int index_md,
                        int index_ic1,
                        int index_ic2,
                        int index_l,
                        int cl_integrand_num_columns,
                        double * cl_integrand,
                        double * cl_integrand_limber,
                        double * primordial_pk,
                        double * transfer_ic1,
                        double * transfer_ic2
                        ) {

  int index_q;
  int index_tt;
  int index_ct;
  int index_d1,index_d2;
  double k;
  double clvalue;
  int index_ic1_ic2;
  double transfer_ic1_temp=0.;
  double transfer_ic2_temp=0.;
  double transfer_ic1_temp_reco=0.;
  double transfer_ic2_temp_reco=0.;
  double transfer_ic1_temp_reio=0.;
  double transfer_ic2_temp_reio=0.;
  double transfer_ic1_e_reco=0.;
  double transfer_ic2_e_reco=0.;
  double transfer_ic1_e_reio=0.;
  double transfer_ic2_e_reio=0.;
  double * cl_integrand_ee_reco = NULL;
  double * cl_integrand_ee_reio = NULL;
  double * cl_integrand_te_reco = NULL;
  double * cl_integrand_te_reio = NULL;
  double cl_ee_reco = 0.0;
  double cl_ee_reio = 0.0;
  double cl_te_reco = 0.0;
  double cl_te_reio = 0.0;
  double * transfer_ic1_nc=NULL;
  double * transfer_ic2_nc=NULL;
  double factor;
  int index_q_spline=0;
  double * integrand;
  int num_columns;
  int num_k;
  int column_k;
  int column_integrand;
  int column_derivative;
  int index_spline;
  double q_min;
  double k_min;
  double l;

  l = phr->l[index_l];

  index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,phr->ic_size[index_md]);

  if (ppt->has_cl_number_count == _TRUE_ && _scalars_) {
    class_alloc(transfer_ic1_nc,phr->d_size*sizeof(double),phr->error_message);
    class_alloc(transfer_ic2_nc,phr->d_size*sizeof(double),phr->error_message);
  }
  if (_scalars_) {
    class_alloc(cl_integrand_ee_reco,3*ptr->q_size*sizeof(double),phr->error_message);
    class_alloc(cl_integrand_ee_reio,3*ptr->q_size*sizeof(double),phr->error_message);
    class_alloc(cl_integrand_te_reco,3*ptr->q_size*sizeof(double),phr->error_message);
    class_alloc(cl_integrand_te_reio,3*ptr->q_size*sizeof(double),phr->error_message);
  }

  /* Technical point: here, we will do a spline integral over the
     whole range of k's, excepted in the closed (K>0) case. In that
     case, it is a bad idea to spline over the values of k
     corresponding to nu<nu_flat_approximation. In this region, nu
     values are integer values, so the steps dq and dk have some
     discrete jumps. This makes the spline routine less accurate than
     a trapezoidal integral with finer sampling. So, in the closed
     case, we set index_q_spline to ptr->index_q_flat_approximation,
     to tell the integration routine that below this index, it should
     treat the integral as a trapezoidal one. For testing, one is free
     to set index_q_spline to 0, to enforce spline integration
     everywhere, or to (ptr->q_size-1), to enforce trapezoidal
     integration everywhere. */

  if (pba->sgnK == 1) {
    index_q_spline = ptr->index_q_flat_approximation;
  }

  for (index_q=0; index_q < ptr->q_size; index_q++) {

    //q = ptr->q[index_q];
    k = ptr->k[index_md][index_q];

    cl_integrand[index_q*cl_integrand_num_columns+0] = k;

    class_call(primordial_spectrum_at_k(ppm,index_md,linear,k,primordial_pk),
               ppm->error_message,
               phr->error_message);

    /* above routine checks that k>0: no possible division by zero below */

    for (index_tt=0; index_tt < ptr->tt_size[index_md]; index_tt++) {

      transfer_ic1[index_tt] =
        ptr->transfer[index_md]
        [((index_ic1 * ptr->tt_size[index_md] + index_tt)
          * ptr->l_size[index_md] + index_l)
         * ptr->q_size + index_q];

      if (index_ic1 == index_ic2) {
        transfer_ic2[index_tt] = transfer_ic1[index_tt];
      }
      else {
        transfer_ic2[index_tt] = ptr->transfer[index_md]
          [((index_ic2 * ptr->tt_size[index_md] + index_tt)
            * ptr->l_size[index_md] + index_l)
           * ptr->q_size + index_q];
      }
    }

    /* define combinations of transfer functions */

    if (ppt->has_cl_cmb_temperature == _TRUE_) {

      if (_scalars_) {

        transfer_ic1_temp = transfer_ic1[ptr->index_tt_t0] + transfer_ic1[ptr->index_tt_t1] + transfer_ic1[ptr->index_tt_t2];
        transfer_ic2_temp = transfer_ic2[ptr->index_tt_t0] + transfer_ic2[ptr->index_tt_t1] + transfer_ic2[ptr->index_tt_t2];

        /*
         * For the tomographic C_l^{alpha T} pieces, use the epoch-paired
         * temperature transfers rather than the full temperature source.
         * This matches the reco/reio decomposition used for alpha and E.
         */
        transfer_ic1_temp_reco =
          transfer_ic1[ptr->index_tt_t0_reco] +
          transfer_ic1[ptr->index_tt_t1_reco] +
          transfer_ic1[ptr->index_tt_t2_reco];
        transfer_ic2_temp_reco =
          transfer_ic2[ptr->index_tt_t0_reco] +
          transfer_ic2[ptr->index_tt_t1_reco] +
          transfer_ic2[ptr->index_tt_t2_reco];

        transfer_ic1_temp_reio =
          transfer_ic1[ptr->index_tt_t0_reio] +
          transfer_ic1[ptr->index_tt_t1_reio] +
          transfer_ic1[ptr->index_tt_t2_reio];
        transfer_ic2_temp_reio =
          transfer_ic2[ptr->index_tt_t0_reio] +
          transfer_ic2[ptr->index_tt_t1_reio] +
          transfer_ic2[ptr->index_tt_t2_reio];

      }

      if (_vectors_) {

        transfer_ic1_temp = transfer_ic1[ptr->index_tt_t1] + transfer_ic1[ptr->index_tt_t2];
        transfer_ic2_temp = transfer_ic2[ptr->index_tt_t1] + transfer_ic2[ptr->index_tt_t2];

      }

      if (_tensors_) {

        transfer_ic1_temp = transfer_ic1[ptr->index_tt_t2];
        transfer_ic2_temp = transfer_ic2[ptr->index_tt_t2];

      }
    }

    if (ppt->has_cl_cmb_polarization == _TRUE_) {

      if (_scalars_) {
        transfer_ic1_e_reco = transfer_ic1[ptr->index_tt_e_reco];
        transfer_ic2_e_reco = transfer_ic2[ptr->index_tt_e_reco];

        transfer_ic1_e_reio = transfer_ic1[ptr->index_tt_e_reio];
        transfer_ic2_e_reio = transfer_ic2[ptr->index_tt_e_reio];
      }
    }

    if (ppt->has_cl_number_count == _TRUE_ && _scalars_) {

      for (index_d1=0; index_d1<phr->d_size; index_d1++) {

        transfer_ic1_nc[index_d1] = 0.;
        transfer_ic2_nc[index_d1] = 0.;

        if (ppt->has_nc_density == _TRUE_) {
          transfer_ic1_nc[index_d1] += transfer_ic1[ptr->index_tt_density+index_d1];
          transfer_ic2_nc[index_d1] += transfer_ic2[ptr->index_tt_density+index_d1];
        }

        if (ppt->has_nc_rsd     == _TRUE_) {
          transfer_ic1_nc[index_d1]
            += transfer_ic1[ptr->index_tt_rsd+index_d1]
            + transfer_ic1[ptr->index_tt_d0+index_d1]
            + transfer_ic1[ptr->index_tt_d1+index_d1];
          transfer_ic2_nc[index_d1]
            += transfer_ic2[ptr->index_tt_rsd+index_d1]
            + transfer_ic2[ptr->index_tt_d0+index_d1]
            + transfer_ic2[ptr->index_tt_d1+index_d1];
        }

        if (ppt->has_nc_lens == _TRUE_) {
          transfer_ic1_nc[index_d1] +=
            l*(l+1.)*transfer_ic1[ptr->index_tt_nc_lens+index_d1];
          transfer_ic2_nc[index_d1] +=
            l*(l+1.)*transfer_ic2[ptr->index_tt_nc_lens+index_d1];
        }

        if (ppt->has_nc_gr == _TRUE_) {
          transfer_ic1_nc[index_d1]
            += transfer_ic1[ptr->index_tt_nc_g1+index_d1]
            + transfer_ic1[ptr->index_tt_nc_g2+index_d1]
            + transfer_ic1[ptr->index_tt_nc_g3+index_d1]
            + transfer_ic1[ptr->index_tt_nc_g4+index_d1]
            + transfer_ic1[ptr->index_tt_nc_g5+index_d1];
          transfer_ic2_nc[index_d1]
            += transfer_ic2[ptr->index_tt_nc_g1+index_d1]
            + transfer_ic2[ptr->index_tt_nc_g2+index_d1]
            + transfer_ic2[ptr->index_tt_nc_g3+index_d1]
            + transfer_ic2[ptr->index_tt_nc_g4+index_d1]
            + transfer_ic2[ptr->index_tt_nc_g5+index_d1];
        }

      }
    }

    /* integrand of Cl's */

    /* note: we must integrate

       C_l = int [4 pi dk/k calP(k) Delta1_l(q) Delta2_l(q)]

       where calP(k) is the dimensionless
       power spectrum equal to a constant in the scale-invariant case,
       and to P(k) = A_s k^(ns-1) otherwise and q=sqrt(k2+K) (scalars)
       or sqrt(k2+2K) (vectors) or sqrt(k2+3K) (tensors)

       In the literature, people often rewrite the integral in terms
       of q and absorb the Jacobian of the change of variables in a redefinition of the primodial
       spectrum. Let us illustrate this for scalars:

       dk/k = kdk/k2 = qdq/k2 = dq/q * (q/k)^2 = dq/q * [q2/(q2-K)] = q2dq * 1/[q(q2-K)]

       This factor 1/[q(q2-K)] is commonly absorbed in the definition of calP. Then one would have

       C_l = int [4 pi q2 dq {A_s k^(ns-1)/[q(q2-K)]} Delta1_l(q) Delta2_l(q)]

       Sometimes in the literature, the factor (k2-3K)=(q2-4K) present
       in the initial conditions of scalar transfer functions (if
       normalized to curvature R=1) is also absorbed in the definition
       of the power spectrum. Then the curvature power spectrum reads

       calP = (q2-4K)/[q(q2-K)] * (k/k)^ns

       In CLASS we prefer to define calP = (k/k)^ns like in the flat
       case, to have the factor (q2-4K) in the initialk conditions,
       and the factor 1/[q(q2-K)] doesn't need to be there since we
       integrate over dk/k.

       For tensors, the change of variable described above gives a slightly different result:

       dk/k = kdk/k2 = qdq/k2 = dq/q * (q/k)^2 = dq/q * [q2/(q2-3K)] = q2dq * 1/[q(q2-3K)]

       But for tensors there are extra curvature-related correction factors to
       take into account. See the comments in the perturbation module,
       related to initial conditions for tensors.

    */

    factor = 4. * _PI_ / k;
    if (_scalars_) {

      /* EE reco block: column 0 = k, column 1 = value, column 2 = spline derivative */
      cl_integrand_ee_reco[3*index_q + 0] = k;
      cl_integrand_ee_reco[3*index_q + 1] =
        primordial_pk[index_ic1_ic2]
        * transfer_ic1_e_reco
        * transfer_ic2_e_reco
        * factor;
      cl_integrand_ee_reco[3*index_q + 2] = 0.0;

      /* EE reio block */
      cl_integrand_ee_reio[3*index_q + 0] = k;
      cl_integrand_ee_reio[3*index_q + 1] =
        primordial_pk[index_ic1_ic2]
        * transfer_ic1_e_reio
        * transfer_ic2_e_reio
        * factor;
      cl_integrand_ee_reio[3*index_q + 2] = 0.0;

      /* TE reco block */
      cl_integrand_te_reco[3*index_q + 0] = k;
      cl_integrand_te_reco[3*index_q + 1] =
        primordial_pk[index_ic1_ic2]
        * 0.5 * (transfer_ic1_temp_reco * transfer_ic2_e_reco
               + transfer_ic1_e_reco    * transfer_ic2_temp_reco)
        * factor;
      cl_integrand_te_reco[3*index_q + 2] = 0.0;

      /* TE reio block */
      cl_integrand_te_reio[3*index_q + 0] = k;
      cl_integrand_te_reio[3*index_q + 1] =
        primordial_pk[index_ic1_ic2]
        * 0.5 * (transfer_ic1_temp_reio * transfer_ic2_e_reio
               + transfer_ic1_e_reio    * transfer_ic2_temp_reio)
        * factor;
      cl_integrand_te_reio[3*index_q + 2] = 0.0;
    }

    if (phr->has_tt == _TRUE_)
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_tt]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1_temp
        * transfer_ic2_temp
        * factor;

    if (phr->has_ee == _TRUE_)
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ee]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_e]
        * transfer_ic2[ptr->index_tt_e]
        * factor;

    if (phr->has_te == _TRUE_)
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_te]=
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1_temp * transfer_ic2[ptr->index_tt_e] +
               transfer_ic1[ptr->index_tt_e] * transfer_ic2_temp)
        * factor;

    if (_tensors_ && (phr->has_bb == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_bb]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_b]
        * transfer_ic2[ptr->index_tt_b]
        * factor;

    if (_scalars_ && (phr->has_pp == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_pp]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_lcmb]
        * transfer_ic2[ptr->index_tt_lcmb]
        * factor;

    if (_scalars_ && (phr->has_tp == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_tp]=
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1_temp * transfer_ic2[ptr->index_tt_lcmb] +
               transfer_ic1[ptr->index_tt_lcmb] * transfer_ic2_temp)
        * factor;

    if (_scalars_ && (phr->has_ep == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ep]=
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_e] * transfer_ic2[ptr->index_tt_lcmb] +
               transfer_ic1[ptr->index_tt_lcmb] * transfer_ic2[ptr->index_tt_e])
        * factor;

    if (_scalars_ && (phr->has_aa == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_aa]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_alpha]
        * transfer_ic2[ptr->index_tt_alpha]
        * factor;

    if (_scalars_ && (phr->has_aa_reco == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_aa_reco]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_alpha_reco]
        * transfer_ic2[ptr->index_tt_alpha_reco]
        * factor;

    if (_scalars_ && (phr->has_aa_reio == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_aa_reio]=
        primordial_pk[index_ic1_ic2]
        * transfer_ic1[ptr->index_tt_alpha_reio]
        * transfer_ic2[ptr->index_tt_alpha_reio]
        * factor;

    /* alpha-T cross spectrum */
    if (_scalars_ && (phr->has_at == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_at] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha] * transfer_ic2_temp +
              transfer_ic1_temp * transfer_ic2[ptr->index_tt_alpha])
        * factor;

    /* alpha-T reco */
    if (_scalars_ && (phr->has_at_reco == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_at_reco] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha_reco] * transfer_ic2_temp +
              transfer_ic1_temp * transfer_ic2[ptr->index_tt_alpha_reco])
        * factor;

    /* alpha-T reio */
    if (_scalars_ && (phr->has_at_reio == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_at_reio] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha_reio] * transfer_ic2_temp +
              transfer_ic1_temp * transfer_ic2[ptr->index_tt_alpha_reio])
        * factor;

    /* alpha-E cross spectrum */
    if (_scalars_ && (phr->has_ae == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ae] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha] * transfer_ic2[ptr->index_tt_e] +
              transfer_ic1[ptr->index_tt_e] * transfer_ic2[ptr->index_tt_alpha])
        * factor;

    /* alpha-E reco */
    if (_scalars_ && (phr->has_ae_reco == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ae_reco] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha_reco] * transfer_ic2_e_reco +
              transfer_ic1_e_reco * transfer_ic2[ptr->index_tt_alpha_reco])
        * factor;

    /* alpha-E reio */
    if (_scalars_ && (phr->has_ae_reio == _TRUE_))
      cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ae_reio] =
        primordial_pk[index_ic1_ic2]
        * 0.5*(transfer_ic1[ptr->index_tt_alpha_reio] * transfer_ic2_e_reio +
              transfer_ic1_e_reio * transfer_ic2[ptr->index_tt_alpha_reio])
        * factor;

    if (_scalars_ && (phr->has_dd == _TRUE_)) {
      index_ct=0;
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        for (index_d2=index_d1; index_d2<=MIN(index_d1+phr->non_diag,phr->d_size-1); index_d2++) {
          cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_dd+index_ct]=
            primordial_pk[index_ic1_ic2]
            * transfer_ic1_nc[index_d1]
            * transfer_ic2_nc[index_d2]
            * factor;
          index_ct++;
        }
      }
    }

    if (_scalars_ && (phr->has_td == _TRUE_)) {
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_td+index_d1]=
          primordial_pk[index_ic1_ic2]
          * 0.5*(transfer_ic1_temp * transfer_ic2_nc[index_d1] +
                 transfer_ic1_nc[index_d1] * transfer_ic2_temp)
          * factor;
      }
    }

    if (_scalars_ && (phr->has_pd == _TRUE_)) {
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_pd+index_d1]=
          primordial_pk[index_ic1_ic2]
          * 0.5*(transfer_ic1[ptr->index_tt_lcmb] * transfer_ic2_nc[index_d1] +
                 transfer_ic1_nc[index_d1] * transfer_ic2[ptr->index_tt_lcmb])
          * factor;
      }
    }

    if (_scalars_ && (phr->has_ll == _TRUE_)) {
      index_ct=0;
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        for (index_d2=index_d1; index_d2<=MIN(index_d1+phr->non_diag,phr->d_size-1); index_d2++) {
          cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_ll+index_ct]=
            primordial_pk[index_ic1_ic2]
            * transfer_ic1[ptr->index_tt_lensing+index_d1]
            * transfer_ic2[ptr->index_tt_lensing+index_d2]
            * factor;
          index_ct++;
        }
      }
    }

    if (_scalars_ && (phr->has_tl == _TRUE_)) {
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_tl+index_d1]=
          primordial_pk[index_ic1_ic2]
          * 0.5*(transfer_ic1_temp * transfer_ic2[ptr->index_tt_lensing+index_d1] +
                 transfer_ic1[ptr->index_tt_lensing+index_d1] * transfer_ic2_temp)
          * factor;
      }
    }

    if (_scalars_ && (phr->has_dl == _TRUE_)) {
      index_ct=0;
      for (index_d1=0; index_d1<phr->d_size; index_d1++) {
        for (index_d2=MAX(index_d1-phr->non_diag,0); index_d2<=MIN(index_d1+phr->non_diag,phr->d_size-1); index_d2++) {
          cl_integrand[index_q*cl_integrand_num_columns+1+phr->index_ct_dl+index_ct]=
            primordial_pk[index_ic1_ic2]
            * transfer_ic1_nc[index_d1] * transfer_ic2[ptr->index_tt_lensing+index_d2]
            * factor;
          index_ct++;
        }
      }
    }
  }
  if (_scalars_) {

    q_min = ptr->q[0];
    k_min = ptr->k[index_md][0];

    class_call(array_spline(cl_integrand_ee_reco,
                            3,
                            ptr->q_size,
                            0,
                            1,
                            2,
                            _SPLINE_EST_DERIV_,
                            phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_integrate_all_trapzd_or_spline(cl_integrand_ee_reco,
                                                    3,
                                                    ptr->q_size,
                                                    index_q_spline,
                                                    0,
                                                    1,
                                                    2,
                                                    &cl_ee_reco,
                                                    phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_spline(cl_integrand_ee_reio,
                            3,
                            ptr->q_size,
                            0,
                            1,
                            2,
                            _SPLINE_EST_DERIV_,
                            phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_integrate_all_trapzd_or_spline(cl_integrand_ee_reio,
                                                    3,
                                                    ptr->q_size,
                                                    index_q_spline,
                                                    0,
                                                    1,
                                                    2,
                                                    &cl_ee_reio,
                                                    phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_spline(cl_integrand_te_reco,
                            3,
                            ptr->q_size,
                            0,
                            1,
                            2,
                            _SPLINE_EST_DERIV_,
                            phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_integrate_all_trapzd_or_spline(cl_integrand_te_reco,
                                                    3,
                                                    ptr->q_size,
                                                    index_q_spline,
                                                    0,
                                                    1,
                                                    2,
                                                    &cl_te_reco,
                                                    phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_spline(cl_integrand_te_reio,
                            3,
                            ptr->q_size,
                            0,
                            1,
                            2,
                            _SPLINE_EST_DERIV_,
                            phr->error_message),
               phr->error_message,
               phr->error_message);

    class_call(array_integrate_all_trapzd_or_spline(cl_integrand_te_reio,
                                                    3,
                                                    ptr->q_size,
                                                    index_q_spline,
                                                    0,
                                                    1,
                                                    2,
                                                    &cl_te_reio,
                                                    phr->error_message),
               phr->error_message,
               phr->error_message);

    if (pba->sgnK == 1) {
      cl_ee_reco += cl_integrand_ee_reco[1] * q_min/k_min*sqrt(pba->K)/2.;
      cl_ee_reio += cl_integrand_ee_reio[1] * q_min/k_min*sqrt(pba->K)/2.;
      cl_te_reco += cl_integrand_te_reco[1] * q_min/k_min*sqrt(pba->K)/2.;
      cl_te_reio += cl_integrand_te_reio[1] * q_min/k_min*sqrt(pba->K)/2.;
    }

    harmonic_ee_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2] = cl_ee_reco;
    harmonic_ee_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2] = cl_ee_reio;
    harmonic_te_reco_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2] = cl_te_reco;
    harmonic_te_reio_block[index_md][index_l * phr->ic_ic_size[index_md] + index_ic1_ic2] = cl_te_reio;
  }

  /* do also a full limber calculation for some types (actually, only pp) */

  if ((ptr->do_lcmb_full_limber == _TRUE_)  && (l>ppr->l_switch_limber)) {

    for (index_q=0; index_q < ptr->q_size_limber; index_q++) {

      //q = ptr->q_limber[index_q];
      k = ptr->k_limber[index_md][index_q];

      cl_integrand_limber[index_q*cl_integrand_num_columns+0] = k;

      class_call(primordial_spectrum_at_k(ppm,index_md,linear,k,primordial_pk),
                 ppm->error_message,
                 phr->error_message);

      /* This is where we define for which types of Cl's we want a
         full Limber version. If we wanted it for more than phiphi, we
         would add other if statements below. */

      if (_scalars_ && (phr->has_pp == _TRUE_)) {

        index_tt = ptr->index_tt_lcmb;
        index_ct = phr->index_ct_pp;

        transfer_ic1[index_tt] =
          ptr->transfer_limber[index_md]
          [((index_ic1 * ptr->tt_size[index_md] + ptr->index_tt_lcmb)
            * ptr->l_size[index_md] + index_l)
           * ptr->q_size_limber + index_q];

        if (index_ic1 == index_ic2) {
          transfer_ic2[index_tt] = transfer_ic1[ptr->index_tt_lcmb];
        }
        else {
          transfer_ic2[index_tt] = ptr->transfer_limber[index_md]
            [((index_ic2 * ptr->tt_size[index_md] + ptr->index_tt_lcmb)
              * ptr->l_size[index_md] + index_l)
             * ptr->q_size_limber + index_q];
        }

        factor = 4. * _PI_ / k;

        cl_integrand_limber[index_q*cl_integrand_num_columns+1+phr->index_ct_pp]=
          primordial_pk[index_ic1_ic2]
          * transfer_ic1[index_tt]
          * transfer_ic2[index_tt]
          * factor;
      }
    }
  }

  for (index_ct=0; index_ct<phr->ct_size; index_ct++) {

    /* treat null spectra (C_l^BB of scalars, C_l^pp of tensors, etc. */

    if ((_scalars_ && (phr->has_bb == _TRUE_) && (index_ct == phr->index_ct_bb)) ||
        (_tensors_ && (phr->has_pp == _TRUE_) && (index_ct == phr->index_ct_pp)) ||
        (_tensors_ && (phr->has_tp == _TRUE_) && (index_ct == phr->index_ct_tp)) ||
        (_tensors_ && (phr->has_ep == _TRUE_) && (index_ct == phr->index_ct_ep)) ||
        (_tensors_ && (phr->has_dd == _TRUE_) && (index_ct == phr->index_ct_dd)) ||
        (_tensors_ && (phr->has_td == _TRUE_) && (index_ct == phr->index_ct_td)) ||
        (_tensors_ && (phr->has_pd == _TRUE_) && (index_ct == phr->index_ct_pd)) ||
        (_tensors_ && (phr->has_ll == _TRUE_) && (index_ct == phr->index_ct_ll)) ||
        (_tensors_ && (phr->has_tl == _TRUE_) && (index_ct == phr->index_ct_tl)) ||
        (_tensors_ && (phr->has_dl == _TRUE_) && (index_ct == phr->index_ct_dl)) ||
        (_tensors_ && (phr->has_aa == _TRUE_) && (index_ct == phr->index_ct_aa)) ||
        (_tensors_ && (phr->has_at == _TRUE_) && (index_ct == phr->index_ct_at)) ||
        (_tensors_ && (phr->has_ae == _TRUE_) && (index_ct == phr->index_ct_ae))
        ) {

      phr->cl[index_md]
        [(index_l * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size + index_ct] = 0.;

    }
    /* for non-zero spectra, integrate over q */
    else {

      /* spline the integrand over the whole range of k's. This is
         where we decide which of the normal or full Limber scheme
         will be used at the end. */

      if (_scalars_ && (ptr->do_lcmb_full_limber == _TRUE_) && (phr->has_pp == _TRUE_) && (index_ct == phr->index_ct_pp) && (l>ppr->l_switch_limber)) {
        integrand = cl_integrand_limber;
        num_columns = cl_integrand_num_columns;
        num_k = ptr->q_size_limber;
        index_spline = 0;
        q_min = ptr->q_limber[0];
        k_min = ptr->k_limber[0][0];
      }
      else{
        integrand = cl_integrand;
        num_columns = cl_integrand_num_columns;
        num_k = ptr->q_size;
        index_spline = index_q_spline;
        q_min = ptr->q[0];
        k_min = ptr->k[index_md][0];
      }

      column_k = 0;
      column_integrand = 1+index_ct;
      column_derivative = 1+phr->ct_size+index_ct;

      class_call(array_spline(integrand,
                              num_columns,
                              num_k,
                              column_k,
                              column_integrand,
                              column_derivative,
                              _SPLINE_EST_DERIV_,
                              phr->error_message),
                 phr->error_message,
                 phr->error_message);

      class_call(array_integrate_all_trapzd_or_spline(integrand,
                                                      num_columns,
                                                      num_k,
                                                      index_spline,
                                                      column_k,
                                                      column_integrand,
                                                      column_derivative,
                                                      &clvalue,
                                                      phr->error_message),
                 phr->error_message,
                 phr->error_message);

      /* in the closed case, instead of an integral, we have a
         discrete sum. In practice, this does not matter: the previous
         routine does give a correct approximation of the discrete
         sum, both in the trapezoidal and spline regions. The only
         error comes from the first point: the previous routine
         assumes a weight for the first point which is too small
         compared to what it would be in the an actual discrete
         sum. The line below correct this problem in an exact way.
      */

      if (pba->sgnK == 1) {
        clvalue += integrand[1+index_ct] * q_min/k_min*sqrt(pba->K)/2.;
      }

      /* we have the correct C_l now. We can store it in the transfer structure. */

      phr->cl[index_md]
        [(index_l * phr->ic_ic_size[index_md] + index_ic1_ic2) * phr->ct_size + index_ct]
        = clvalue;

    }
  }

  if (ppt->has_cl_number_count == _TRUE_ && _scalars_) {
    free(transfer_ic1_nc);
    free(transfer_ic2_nc);
  }
  if (_scalars_) {
    free(cl_integrand_ee_reco);
    free(cl_integrand_ee_reio);
    free(cl_integrand_te_reco);
    free(cl_integrand_te_reio);
  }

  return _SUCCESS_;

}

/* deprecated functions (since v2.8) */

/**
 * Matter power spectrum for arbitrary redshift and for all initial conditions.
 *
 * This function is deprecated since v2.8. Try using fourier_pk_at_z() instead.
 *
 * @param pba           Input: pointer to background structure (used for converting z into tau)
 * @param phr           Input: pointer to harmonic structure (containing pre-computed table)
 * @param mode          Input: linear or logarithmic
 * @param z             Input: redshift
 * @param output_tot    Output: total matter power spectrum P(k) in \f$ Mpc^3 \f$ (linear mode), or its logarithms (logarithmic mode)
 * @param output_ic     Output: for each pair of initial conditions, matter power spectra P(k) in \f$ Mpc^3 \f$ (linear mode), or their logarithms and cross-correlation angles (logarithmic mode)
 * @param output_cb_tot Output: CDM+baryon power spectrum P_cb(k) in \f$ Mpc^3 \f$ (linear mode), or its logarithms (logarithmic mode)
 * @param output_cb_ic  Output: for each pair of initial conditions, CDM+baryon power spectra P_cb(k) in \f$ Mpc^3 \f$ (linear mode), or their logarithms and cross-correlation angles (logarithmic mode)
 * @return the error status
 */

int harmonic_pk_at_z(
                     struct background * pba,
                     struct harmonic * phr,
                     enum linear_or_logarithmic mode,
                     double z,
                     double * output_tot,    /* array with argument output_tot[index_k] (must be already allocated) */
                     double * output_ic,     /* array with argument output_tot[index_k * phr->ic_ic_size[index_md] + index_ic1_ic2] (must be already allocated only if more than one initial condition) */
                     double * output_cb_tot, /* same as output_tot for the baryon+CDM only */
                     double * output_cb_ic   /* same as output_ic  for the baryon+CDM only */
                     ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_pk_at_z() which is deprecated since v2.8. It will soon be removed. Use fourier_pk_at_z() instead.\n");

  class_call(fourier_pks_at_z(
                              pba,
                              phr->pfo,
                              mode,
                              pk_linear,
                              z,
                              output_tot,
                              output_ic,
                              output_cb_tot,
                              output_cb_ic
                              ),
             phr->pfo->error_message,
             phr->error_message);

  return _SUCCESS_;

}

/**
 * Matter power spectrum for arbitrary wavenumber, redshift and initial condition.
 *
 * This function is deprecated since v2.8. Try using fourier_pk_linear_at_k_and_z() instead.
 *
 * @param pba        Input: pointer to background structure (used for converting z into tau)
 * @param ppm        Input: pointer to primordial structure (used only in the case 0 < k < kmin)
 * @param phr        Input: pointer to harmonic structure (containing pre-computed table)
 * @param k          Input: wavenumber in 1/Mpc
 * @param z          Input: redshift
 * @param pk_tot     Output: total matter power spectrum P(k) in \f$ Mpc^3 \f$
 * @param pk_ic      Output: for each pair of initial conditions, matter power spectra P(k) in \f$ Mpc^3\f$
 * @param pk_cb_tot  Output: b+CDM power spectrum P(k) in \f$ Mpc^3 \f$
 * @param pk_cb_ic   Output: for each pair of initial conditions, b+CDM power spectra P(k) in \f$ Mpc^3\f$
 * @return the error status
 */

int harmonic_pk_at_k_and_z(
                           struct background * pba,
                           struct primordial * ppm,
                           struct harmonic * phr,
                           double k,
                           double z,
                           double * pk_tot,    /* pointer to a single number (must be already allocated) */
                           double * pk_ic,     /* array of argument pk_ic[index_ic1_ic2]
                                                  (must be already allocated only if several initial conditions) */
                           double * pk_cb_tot, /* same as pk_tot for baryon+CDM part only */
                           double * pk_cb_ic   /* same as pk_ic  for baryon+CDM part only */
                           ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_pk_at_k_and_z() which is deprecated since v2.8. It will soon be removed. Use fourier_pk_linear_at_k_and_z() instead.\n");

  class_call(fourier_pks_at_k_and_z(pba,
                                    ppm,
                                    phr->pfo,
                                    pk_linear,
                                    k,
                                    z,
                                    pk_tot,
                                    pk_ic,
                                    pk_cb_tot,
                                    pk_cb_ic),
             phr->pfo->error_message,
             phr->error_message);

  return _SUCCESS_;
}

/**
 * Non-linear total matter power spectrum for arbitrary redshift.
 *
 * This function is deprecated since v2.8. Try using fourier_pk_at_z() instead.
 *
 * @param pba           Input: pointer to background structure (used for converting z into tau)
 * @param phr           Input: pointer to harmonic structure (containing pre-computed table)
 * @param mode          Input: linear or logarithmic
 * @param z             Input: redshift
 * @param output_tot    Output: total matter power spectrum P(k) in \f$ Mpc^3\f$ (linear mode), or its logarithms (logarithmic mode)
 * @param output_cb_tot Output: b+CDM power spectrum P(k) in \f$ Mpc^3\f$ (linear mode), or its logarithms (logarithmic mode)
 * @return the error status
 */

int harmonic_pk_nl_at_z(
                        struct background * pba,
                        struct harmonic * phr,
                        enum linear_or_logarithmic mode,
                        double z,
                        double * output_tot,   /* array with argument output_tot[index_k] (must be already allocated) */
                        double * output_cb_tot
                        ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_pk_nl_at_z() which is deprecated since v2.8. It will soon be removed. Use fourier_pk_at_z() instead.\n");

  class_call(fourier_pks_at_z(pba,
                              phr->pfo,
                              mode,
                              pk_nonlinear,
                              z,
                              output_tot,
                              NULL,
                              output_cb_tot,
                              NULL
                              ),
             phr->pfo->error_message,
             phr->error_message);

  return _SUCCESS_;

}

/**
 * Non-linear total matter power spectrum for arbitrary wavenumber and redshift.
 *
 * This function is deprecated since v2.8. Try using fourier_pk_at_k_and_z() instead.
 *
 * @param pba        Input: pointer to background structure (used for converting z into tau)
 * @param ppm        Input: pointer to primordial structure (used only in the case 0 < k < kmin)
 * @param phr        Input: pointer to harmonic structure (containing pre-computed table)
 * @param k          Input: wavenumber in 1/Mpc
 * @param z          Input: redshift
 * @param pk_tot     Output: total matter power spectrum P(k) in \f$ Mpc^3\f$
 * @param pk_cb_tot  Output: b+CDM power spectrum P(k) in \f$ Mpc^3\f$
 * @return the error status
 */

int harmonic_pk_nl_at_k_and_z(
                              struct background * pba,
                              struct primordial * ppm,
                              struct harmonic * phr,
                              double k,
                              double z,
                              double * pk_tot,   /* pointer to a single number (must be already allocated) */
                              double * pk_cb_tot /* same as pk_tot for baryon+CDM only */
                              ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_pk_nl_at_k_and_z() which is deprecated since v2.8. It will soon be removed. Use fourier_pk_at_k_and_z() instead.\n");

  class_call(fourier_pks_at_k_and_z(pba,
                                    ppm,
                                    phr->pfo,
                                    pk_nonlinear,
                                    k,
                                    z,
                                    pk_tot,
                                    NULL,
                                    pk_cb_tot,
                                    NULL
                                    ),
             phr->pfo->error_message,
             phr->error_message);

  return _SUCCESS_;

}

/**
 * Return the P(k,z) for a grid of (k_i,z_j) passed in input,
 * for all available pk types (_m, _cb),
 * either linear or nonlinear depending on input.
 *
 * This function is deprecated since v2.8. Try using fourier_pks_at_kvec_and_zvec() instead.
 *
 * @param pba            Input: pointer to background structure
 * @param phr            Input: pointer to harmonic structure
 * @param kvec           Input: array of wavenumbers in ascending order (in 1/Mpc)
 * @param kvec_size      Input: size of array of wavenumbers
 * @param zvec           Input: array of redshifts in arbitrary order
 * @param zvec_size      Input: size of array of redshifts
 * @param pk_tot_out     Output: P(k_i,z_j) for total matter (if available) in Mpc**3
 * @param pk_cb_tot_out  Output: P_cb(k_i,z_j) for cdm+baryons (if available) in Mpc**3
 * @param nonlinear      Input: _TRUE_ or _FALSE_ (to output nonlinear or linear P(k,z))
 * @return the error status
 */

int harmonic_fast_pk_at_kvec_and_zvec(
                                      struct background * pba,
                                      struct harmonic * phr,
                                      double * kvec,
                                      int kvec_size,
                                      double * zvec,
                                      int zvec_size,
                                      double * pk_tot_out, // pk_tot_out[index_zvec*kvec_size+index_kvec],
                                                           // already allocated
                                                           //(or NULL if user knows there is no _m output)
                                      double * pk_cb_tot_out, // idem
                                      int nonlinear
                                      ) {
  enum pk_outputs pk_output;

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_fast_pks_at_kvec_and_zvec() which is deprecated since v2.8. It will soon be removed. Use fourier_pk_at_kvec_and_zvec() instead.\n");

  if (nonlinear == _TRUE_)
    pk_output = pk_nonlinear;
  else
    pk_output = pk_linear;

  class_call(fourier_pks_at_kvec_and_zvec(
                                          pba,
                                          phr->pfo,
                                          pk_output,
                                          kvec,
                                          kvec_size,
                                          zvec,
                                          zvec_size,
                                          pk_tot_out,
                                          pk_cb_tot_out),
             phr->pfo->error_message,
             phr->error_message);

  return _SUCCESS_;
}

/**
 * This routine computes sigma(R) given P(k) for total matter power
 * spectrum (does not check that k_max is large enough)
 *
 * This function is deprecated since v2.8. Try using fourier_sigmas_at_z() instead.
 *
 * @param pba   Input: pointer to background structure
 * @param ppm   Input: pointer to primordial structure
 * @param phr   Input: pointer to harmonic structure
 * @param R     Input: radius in Mpc
 * @param z     Input: redshift
 * @param sigma Output: variance in a sphere of radius R (dimensionless)
 * @return the error status
 */

int harmonic_sigma(
                   struct background * pba,
                   struct primordial * ppm,
                   struct harmonic * phr,
                   double R,
                   double z,
                   double * sigma
                   ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_sigma() which is deprecated since v2.8. It will soon be removed. Use fourier_sigmas_at_z() instead.\n");

  if (phr->pfo->has_pk_m) {

    class_call(fourier_sigma_at_z(pba,
                                  phr->pfo,
                                  R,
                                  z,
                                  phr->pfo->index_pk_m,
                                  80., // hardcoded, yes, but the function is deprecated...
                                  sigma),
               phr->pfo->error_message,
               phr->error_message);

  }

  return _SUCCESS_;
}

/**
 * This routine computes sigma(R) given P(k) for baryon+cdm power
 * spectrum (does not check that k_max is large enough)
 *
 * This function is deprecated since v2.8. Try using fourier_sigmas_at_z() instead.
 *
 * @param pba      Input: pointer to background structure
 * @param ppm      Input: pointer to primordial structure
 * @param phr      Input: pointer to harmonic structure
 * @param R        Input: radius in Mpc
 * @param z        Input: redshift
 * @param sigma_cb Output: variance in a sphere of radius R (dimensionless)
 * @return the error status
 */

int harmonic_sigma_cb(
                      struct background * pba,
                      struct primordial * ppm,
                      struct harmonic * phr,
                      double R,
                      double z,
                      double * sigma_cb
                      ) {

  fprintf(stderr," -> [WARNING:] You are calling the function harmonic_sigma_cb() which is deprecated since v2.8. It will soon be removed. Use fourier_sigmas_at_z() instead.\n");

  if (phr->pfo->has_pk_cb) {

    class_call(fourier_sigma_at_z(pba,
                                  phr->pfo,
                                  R,
                                  z,
                                  phr->pfo->index_pk_cb,
                                  80., // hardcoded, yes, but the function is deprecated...
                                  sigma_cb),
               phr->pfo->error_message,
               phr->error_message);
  }

  return _SUCCESS_;
}

/* deprecated functions (since v2.1) */

/**
 * Obsolete function, superseeded by perturbations_sources_at_tau()
 * (at the time of the switch, this function was anyway never used anywhere)
 *
 * @param pba        Input: pointer to background structure (used for converting z into tau)
 * @param phr        Input: pointer to harmonic structure (containing pre-computed table)
 * @param z          Input: redshift
 * @param output     Output: matter transfer functions
 * @return the error status
 */

int harmonic_tk_at_z(
                     struct background * pba,
                     struct harmonic * phr,
                     double z,
                     double * output /* array with argument output[(index_k*phr->ic_size[index_md]+index_ic)*phr->tr_size+index_tr] (must be already allocated) */
                     ) {


  class_stop(phr->error_message,
             "The function harmonic_tk_at_z() is obsolete, use instead perturbations_sources_at_z(), it does the same");

  return _SUCCESS_;

}

/**
 * Obsolete function, superseeded by perturbations_sources_at_tau()
 * (at the time of the switch, this function was anyway never used anywhere)
 *
 * @param pba        Input: pointer to background structure (used for converting z into tau)
 * @param phr        Input: pointer to harmonic structure (containing pre-computed table)
 * @param k          Input: wavenumber in 1/Mpc
 * @param z          Input: redshift
 * @param output     Output: matter transfer functions
 * @return the error status
 */

int harmonic_tk_at_k_and_z(
                           struct background * pba,
                           struct harmonic * phr,
                           double k,
                           double z,
                           double * output  /* array with argument output[index_ic*phr->tr_size+index_tr] (must be already allocated) */
                           ) {

  class_stop(phr->error_message,
             "The function harmonic_tk_at_k_and_z() is obsolete, use instead perturbations_sources_at_k_and_z(), it does the same");

  return _SUCCESS_;

}

/* end deprecated functions */
