// Builds Counterfactual_Project_2_Results.docx from verified outputs.
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, BorderStyle, PageNumber, Footer } = require('docx');
const sizes = JSON.parse(fs.readFileSync('figs/sizes.json'));
const W = 9360; // 6.5 in content width in DXA
const P = (t, o = {}) => new Paragraph({ spacing: { after: 140, line: 276 }, ...o, children: [new TextRun({ text: t, size: 22 })] });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 140 }, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun(t)] });
const CAP = t => new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: t, size: 19, italics: true, color: '52514E' })] });
function FIG(name, cap, wIn = 6.5) {
  const [w, h] = sizes[name]; const px = wIn * 96;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(`figs/${name}.png`), transformation: { width: px, height: px * h / w } })] }), CAP(cap)];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: 'C9C8C2' };
function TBL(head, rows, widths, cap) {
  const tot = widths.reduce((a, b) => a + b, 0); const ws = widths.map(x => Math.round(x * W / tot));
  ws[ws.length - 1] = W - ws.slice(0, -1).reduce((a, b) => a + b, 0);
  const cell = (t, i, hd) => new TableCell({ width: { size: ws[i], type: WidthType.DXA }, borders: { top: border, bottom: border, left: border, right: border },
    shading: hd ? { type: ShadingType.CLEAR, fill: 'F2F1EC', color: 'auto' } : undefined, margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), size: 18, bold: !!hd })] })] });
  return [CAP(cap), new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: ws,
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, i, true)) }), ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}
const D = JSON.parse(fs.readFileSync('out/report_numbers.json'));
const pct = x => (100 * x).toFixed(1);
const body = [];
body.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun('Counterfactual Project 2')] }));
body.push(P('Monitoring and refreshing stored counterfactual guidance under temporal drift. Methodology and results.'));
body.push(P('Benjamin Gyimah Boateng, University of Maryland. Analysis run 20 October 2026.'));

body.push(H1('1 Summary'));
body.push(P('This report runs the full Project 2 framework on the FRA Class 7 corridor data. The frozen 2013 to 2014 classifier and the 2014 library of 637 counterfactual pairs come from the canonical cache that reproduces the locked paper numbers. Every adaptation method, control, and comparator is evaluated on the same eligible cases, under the same model, with the same direction rule, clipping, and validation contract.'));
body.push(P(`The instance excess benchmark is the strongest adaptation. In the January to April 2016 test window it raises joint success from ${pct(D.js.test.stale)}% under stale transfer to ${pct(D.js.test.inst_margin)}%, against ${pct(D.js.test.fresh)}% for fresh generation. It recovers ${pct(D.rec.test.inst_margin)}% of the gap between stale transfer and fresh generation. Population scalar and diagonal factors selected on development data recover little. Diagonal excess scaling, a registered alternative, reaches ${pct(D.js.test.dg_excess)}% and is the only method that repairs a large share of XLEVEL only defects.`));
body.push(P(`Optimal transport does not add value on this corridor. The best development OT variant is ${pct(D.c4.diff * -1)} points below the best diagonal method in 2016, with a 95% interval of ${pct(-D.c4.hi)} to ${pct(-D.c4.lo)} points below. In the stress tests OT improves on stale transfer under tail inflation, but the instance benchmark is better in every scenario. The multivariate entropic map also distorts actions when nothing changes.`));
body.push(P(`Two findings bear on the main paper. First, the 98.2% reference flip rate is an in sample figure. Applied to 2013 defects, which lie outside the library, stale transfer already reaches only ${pct(D.mon.b13min)}% to ${pct(D.mon.b13max)}% per month. Monthly 2015 validity stays inside that band until June 2015. Second, fresh generation is cheap. It takes ${D.cost.fresh_sec.toFixed(2)} s and about ${Math.round(D.cost.fresh_q)} model queries per case, ${D.cost.fresh_total.toFixed(0)} minutes for all ${D.n.y15 + D.n.y16} eligible 2015 and 2016 cases.`));
body.push(P(`A validation based refresh policy without the support gate matches universal fresh generation in validity and regenerates only ${pct(D.pol.eval_nogate_regen)}% of 2015 evaluation cases and ${pct(D.pol.test_nogate_regen)}% of 2016 cases. The registered support gate is too strict and sends most cases straight to regeneration. Library guidance is not more consistent between inspections than fresh generation, so consistency does not support the stored library either.`));

body.push(H1('2 Data and reference objects'));
body.push(P('The data are 147,550 geometry records at 5,279 locations over 28 monthly inspections from June 2013 to April 2016. The state vector holds GAGE, left and right alignment, WARP, and XLEVEL in inches. Project defect labels use the project limits of the main paper.'));
body.push(P('The frozen classifier, the 2014 library, and the cleaned data are loaded from the canonical cache in vscode/scripts/.cache (model_base_nb.pkl, cf_2014_base_nb.pkl, data.pkl). Session notes identify this cache as the source that reproduces the locked numbers. The executed backup notebook gives slightly different values (P2 of 27.75% and 6.55%), so it is not used. Table 1 compares the reproduction with the locked values.'));
body.push(...TBL(['Quantity', 'Locked value', 'Reproduced here'], D.repro, [4, 2, 2], 'Table 1. Reproduction of locked paper values from the canonical cache.'));
body.push(P('All locked values reproduce. The open item on the 0.0494 versus 0.037 in. WARP perturbation resolves here. The value 0.0494 is the mean absolute WARP perturbation and 0.0374 is the signed mean. Both come from the same 637 pairs. Section 5.6 of the main paper should state that it uses the signed mean.'));

body.push(H1('3 Methodology'));
body.push(H2('3.1 Chronological protocol'));
body.push(P('Each inspection from January 2015 to April 2016 is a target. Factors and maps for a target use only the three most recent completed inspections before it. When a tail or excess statistic has fewer than 20 exceedances, the window expands to earlier inspections. If it is still sparse, the factor shrinks to 1. The reference statistics come from all 2014 geometry, the library year.'));
body.push(P('The analysis uses three periods. January to June 2015 is the development period. It selects the best variant within each ladder level by joint success. July, August, November, and December 2015 form the 2015 evaluation period. January to April 2016 is the final test. No test outcome selects any factor, window, map setting, or policy threshold.'));
body.push(H2('3.2 Eligible cases and the validation contract'));
body.push(P(`Eligible cases are project defects that the frozen classifier also predicts as defective. There are ${D.n.y15} such cases in 2015 and ${D.n.y16} in 2016, the denominators of the locked P2 rates. Every method first applies the paper direction rule to WARP, then clips each component to the FRA outer bounds. Validation checks the resulting state, not the stored pair.`));
body.push(P('Three outcomes are recorded for each case. A model flip means the classifier moves from defective to nondefective. Project compliance means every feature lies within the project limits. Feasibility means the state lies within the FRA bounds intersected with the observed range. Joint success requires all three. Feasible model flip is the primary endpoint and joint success is the co primary engineering diagnostic. The two differ by less than two points for every method, so the text reports joint success.'));
body.push(H2('3.3 Adaptation ladder'));
body.push(P('Retrieval uses the paper P2 rule for every level. Each target takes the action of its nearest 2014 origin in MAD scaled Euclidean distance. Only the transformation of that action changes between levels.'));
body.push(...FIG('fig01_architecture', 'Figure 1. Monitor, adapt, validate, and refresh framework.'));
body.push(...TBL(['Level', 'Variants', 'Definition'], D.ladder, [1.3, 2.2, 5], 'Table 2. Methods. Factors are clipped to [0.5, 8] and the instance factor to [0.25, 8].'));
body.push(P(`The instance margin form scales the WARP action so it covers the target excess plus a margin. The margin is ${D.margin.toFixed(4)} in., the median overshoot beyond the project line in the 2014 library. When a target has no WARP excess, the instance factor is 1.`));
body.push(P('The multivariate OT map uses 649 defective and 1,500 nondefective 2014 states as source atoms, with weights that restore the population defect share. Target atoms come from the calibration window in the same way. The coupling uses log domain Sinkhorn with regularization equal to 5% of the median cost. A barycentric map moves each atom. A Gaussian weighted average of the ten nearest atom displacements extends the map to any state.'));
body.push(H2('3.4 Comparators and controls'));
body.push(P('Fresh generation reruns the paper two stage optimizer with seed 42, λ = 0.5, and the same bounds on every eligible case. Its success counts only when the result passes the same validation. Three controls complete the comparison. P1 applies the signed mean library action. No action applies a zero vector. Threshold correction moves each violated feature to 0.001 in. inside its project limit without using the library.'));
body.push(P('A factorial check uses a model retrained on 2013 to 2015 with the paper hyperparameters. It validates the 2016 library actions and a fresh generation run under that model.'));
body.push(H2('3.5 Statistics'));
body.push(P('Paired differences use a block bootstrap over 200 ft spatial blocks within each mile, with 2,000 resamples. The 2016 test has 13 blocks and four inspections. Sensitivity runs use 100 ft and 500 ft blocks and leave one inspection out. Residual failure uses a binomial model with cluster robust errors by block. Composition standardization reweights group success to the 2014 shares of 78.7% WARP only and 21.3% XLEVEL only defects.'));
body.push(H2('3.6 Stress tests, monitoring, and refresh policy'));
body.push(P('Seven controlled scenarios transform the 2014 geometry. They cover no change, common scaling, heterogeneous scaling, WARP tail inflation, a WARP location shift, a WARP and XLEVEL rotation, and XLEVEL only reweighting. Labels are recomputed and the classifier stays fixed. Calibration uses the transformed population itself, so these runs are transductive by design.'));
body.push(P('Monitoring tracks monthly stale validity, ROC AUC, expected calibration error, WARP spread and exceedance ratios, and the Kolmogorov Smirnov statistic against 2014. Alarm thresholds are fixed before reading 2015 results.'));
body.push(P(`The refresh policy tries the best development variant of each level in order. These are instance, scalar, diagonal, and OT. It accepts the first candidate that passes joint validation. Otherwise it regenerates, and a failed regeneration is unresolved. The registered support gate sends a case straight to regeneration when its reference distance exceeds ${D.sup99.toFixed(2)}, the 99th percentile of leave one out distances within the library. A second version omits the gate.`));
body.push(H2('3.7 Deviations from the proposal'));
body.push(P('Four choices differ from the proposal. The model boundary factor uses the 0.283 in. proxy instead of directional searches. The OT out of sample map is a kernel weighted displacement estimator. Spatial blocks replace a joint block over space and inspection, which four test inspections cannot support. The first run of the factor estimator ignored the registered window and always started from one inspection. That run was corrected and all results were rebuilt. The single inspection behavior survives as the k = 1 sensitivity run.'));

body.push(H1('4 Results'));
body.push(H2('4.1 Monitoring and drift'));
body.push(...FIG('fig02_drift_decomposition', 'Figure 2. Monthly drift statistics relative to 2014 and defect group shares.'));
body.push(P('WARP spread, exceedance frequency, and excess all rise sharply from November 2015. The defect mix also changes. XLEVEL only defects reach 36% to 39% of 2016 cases, and joint and other defects appear for the first time.'));
body.push(...FIG('fig08_monitoring', 'Figure 3. Monthly stale transfer validity and fixed classifier ROC AUC on separate axes.'));
body.push(P(`Validity under the registered 80% alarm fires in January 2015, the first month after the library year. That alarm is misleading. In 2014 the targets are the library origins themselves, so validity there is close to 100% by construction. In 2013, outside the library, monthly validity is ${pct(D.mon.b13min)}% to ${pct(D.mon.b13max)}%. Against that out of library band, validity first falls below in June 2015 and stays below from November 2015. The exceedance ratio alarm fires in April 2015, spread in November 2015, AUC in December 2015, and the KS statistic in January 2016. Calibration error never crosses its threshold.`));
body.push(P('H1 is partly supported. Validity does reveal decay that AUC misses, and AUC never falls below 0.984. Validity does not flag drift earlier than exceedance frequency once the in sample artifact is removed.'));

body.push(H2('4.2 Matched validity'));
body.push(...FIG('fig04_matched_validity', 'Figure 4. Joint success by period for the best development variant of each level, with 95% block bootstrap intervals.'));
body.push(...TBL(['Method', 'Eval 2015', 'Test 2016', 'Full 2015', 'Full 2016'], D.mainrows, [3, 1.3, 1.3, 1.3, 1.3], `Table 3. Joint success (%) on matched eligible cases. Eval 2015 N = ${D.n.eval15}, test 2016 N = ${D.n.y16}.`));
body.push(P(`Stale transfer reproduces the locked collapse. Threshold correction fails almost always, since the classifier boundary sits near 0.283 in., inside the 0.30 in. project limit. The instance margin benchmark recovers ${pct(D.rec.eval.inst_margin)}% of the stale to fresh gap in the 2015 evaluation period and ${pct(D.rec.test.inst_margin)}% in 2016.`));
body.push(...TBL(['Contrast', 'Period', 'Difference (pts)', '95% interval'], D.contr, [3.2, 1.3, 1.3, 1.6], 'Table 4. Prespecified paired contrasts in joint success.'));
body.push(P('H2 is supported for the instance benchmark. Population scalars recover little, as predicted. The development period selected the spread variants. Excess variants do better later but were not selected, since early 2015 showed almost no magnitude drift. Scalar excess reaches 30.4% and diagonal excess 41.3% in 2016.'));
body.push(...FIG('fig03_action_geometry', 'Figure 5. One 2016 WARP only target. Stale and population scaled actions stop short of the classifier boundary.', 6.2));

body.push(H2('4.3 Feature wise adaptation'));
body.push(P(`The diagonal spread variant does not differ from the scalar spread variant in either period. H3 holds in its null form. The XLEVEL excess factor reaches 3.5 to 4.5 in 2016, and diagonal excess raises XLEVEL only joint success to ${pct(D.comp.dg_x)}% against ${pct(D.comp.stale_x)}% for stale transfer. No other method does this. It amplifies the small XLEVEL components already present in matched actions.`));

body.push(H2('4.4 Optimal transport'));
body.push(...FIG('fig07_stress_tests', 'Figure 6. Joint success (%) under controlled transformations of the 2014 geometry.'));
body.push(P('On the real inspections every OT variant stays within a few points of stale transfer. The quantile maps move origins and endpoints by similar amounts, so their difference changes little. Population spread grows by about 1.5 times while the required WARP correction grows by about 3 times. OT inherits the limits of the spread factors.'));
body.push(P('The stress tests show where OT can help. Under WARP tail inflation, the 1D and marginal maps raise joint success from 24% to 61%. They do not help under rotation, location shift, or reweighting, and the marginal map collapses under XLEVEL reweighting. The multivariate map gives 18% with no change at all, so the entropic coupling and out of sample extension bias actions even when the populations match. H4 is not supported. Retaining OT in the ladder costs about 33 s per inspection and adds nothing to the policy.'));

body.push(H2('4.5 Residual failures'));
body.push(...FIG('fig05_residual_failures', 'Figure 7. Failure after all ladder levels by defect group and reference distance. mis is the share with support mismatch.'));
body.push(P('Residual failure concentrates by defect group. WARP only defects fail in 3% of evaluation cases and 9% of test cases. XLEVEL only defects fail in 71% and 85%. Joint and other defects always fail. The library contains no such origins, and their violated features lie outside the support of every matched action. Among WARP only and XLEVEL only cases, failure rises with reference distance (odds ratio 10.4 per log unit) and support mismatch (odds ratio 27.0), with cluster robust p below 0.01. WARP excess has no independent effect once these terms enter. These are associations, not physical mechanisms.'));

body.push(H2('4.6 Refresh policy, consistency, and cost'));
body.push(...FIG('fig06_strategy_comparison', 'Figure 8. Joint success and regeneration share by strategy.'));
body.push(...FIG('fig09_policy_outcomes', 'Figure 9. Which step resolves each case under the registered gate and without it.', 6.0));
body.push(P(`The policy without the gate matches universal fresh generation. It reaches ${pct(D.pol.eval_nogate)}% and ${pct(D.pol.test_nogate)}% joint success against ${pct(D.pol.eval_fresh)}% and ${pct(D.pol.test_fresh)}% for fresh generation. It regenerates ${pct(D.pol.eval_nogate_regen)}% and ${pct(D.pol.test_nogate_regen)}% of cases and uses 41% and 69% of the model queries of universal fresh generation. The instance level resolves almost all accepted cases. Removing OT changes success by less than 0.1 points. The registered gate sends 85% to 97% of cases straight to regeneration, so its threshold is far too strict for this library.`));
body.push(P(`Consistency does not favor the library. Between consecutive inspections at the same location, the mean action change is ${D.cons.stale.toFixed(2)} MAD units for stale transfer, ${D.cons.inst.toFixed(2)} for the instance benchmark, ${D.cons.fresh.toFixed(2)} for fresh generation, and ${D.cons.policy.toFixed(2)} for the policy. Two fresh runs with different seeds give a median difference of ${D.cons.seed_med.toFixed(3)} MAD units. H6 is supported for validity and cost but not for consistency.`));
body.push(...TBL(['Item', 'Value'], D.costrows, [5, 2], 'Table 5. Measured computation on a two core cloud workspace.'));

body.push(H2('4.7 Updated model and sensitivity'));
body.push(P(`Under the model retrained on 2013 to 2015, library actions keep similar 2016 validity (instance ${pct(D.fac.inst_upd)}%, stale ${pct(D.fac.stale_upd)}%). Fresh endpoints generated for the frozen model fall to ${pct(D.fac.fresh_f0_on_upd)}% under the new model. Fresh generation under the new model reaches ${pct(D.fac.fresh_upd)}%. Its WARP driven flip rate is ${pct(D.fac.fresh_upd_wd)}%, matching the locked 84.2%. Window length matters most among the sensitivity settings. A one inspection window gives the best excess factors and a six inspection window the worst. Factor caps of 2 cut 2016 excess results by 6 to 16 points. The tail quantile level changes results by under 2 points.`));

body.push(H1('5 Hypothesis verdicts'));
body.push(...TBL(['Hypothesis', 'Verdict', 'Evidence'], D.verdicts, [1.4, 1.6, 6], 'Table 6. Summary of hypothesis tests.'));

body.push(H1('6 Implications for the main paper'));
body.push(P('No locked number changes. Two interpretive points need attention in the TRB revision. The 98.2% reference rate measures transfer to the library origins themselves. The comparison with 28.5% in 2015 therefore mixes drift with the loss from leaving the library. The 2013 out of library rate, about one third, is the fairer baseline. The decline below that baseline from late 2015 is the drift signal. The second point concerns Section 5.6, which should state that 0.037 in. is the signed mean perturbation.'));
body.push(P('The results also favor a shorter Project 2 paper built on the instance benchmark, the coverage diagnosis, and the gate free refresh policy. OT is better reported as a tested and rejected extension than as a core level.'));

body.push(H1('7 Limitations'));
body.push(P('The corridor has 28 inspections and 13 test blocks, so intervals are wide. The development window saw little drift, which made the selected population factors weak. Joint success counts model flips and project compliance, not realized repairs. The OT estimator is one of several possible choices. A different out of sample map could reduce its no change bias. Stress tests are statistical perturbations, not deterioration models.'));

body.push(H1('8 Open items'));
['Decide whether the main paper adds the 2013 out of library baseline next to the 98.2% reference rate.',
 'Correct Section 5.6 to call 0.037 in. the signed mean perturbation.',
 'Recalibrate the support gate on development data, for example by targeting a regeneration share instead of a library percentile.',
 'Decide whether OT stays in the Project 2 paper as a negative result or moves to a separate note.',
 'Review the recourse robustness literature before positioning. No citations were added.',
 'Consider registering excess factors ahead of spread factors, since development data cannot select them before drift appears.'].forEach(t => body.push(new Paragraph({ bullet: { level: 0 }, spacing: { after: 80 }, children: [new TextRun({ text: t, size: 22 })] })));

body.push(H1('Appendix A Folder contents'));
body.push(...TBL(['Path', 'Contents'], D.files, [2.6, 5], 'Table A1. Files in the counterfactual 2 folder.'));

const doc = new Document({
  styles: { default: { document: { run: { font: 'Calibri' } } },
    paragraphStyles: [{ id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 30, bold: true, color: '1F3864' } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 25, bold: true, color: '2A4D8F' } }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: body }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync('Counterfactual_Project_2_Results.docx', b); console.log('written'); });
