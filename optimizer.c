// optimizer.c
// Dynamic batch-size + ratio optimizer for ./split_image_blur
//
// Strategy:
// 1) Coarse scan batch sizes: 100, 200, 300, ... until performance stops improving
//    (stops after N consecutive "worse" steps by a margin) or a run fails.
// 2) For each batch size:
//    - Calibrate ratio CALIB_ITERS times (each run prints "Recommended GPU ratio: X%")
//      and use the AVERAGE recommended ratio (clamped).
//    - Measure final performance RUNS_PER_CONFIG times with that ratio (average time/imbalance).
// 3) Refinement around best batch: try best±50, best±25, best±10 (valid positives).
//
// Build:  gcc -O2 -std=c11 -Wall -Wextra -o optimizer optimizer.c
// Run:    ./optimizer
//         ./optimizer ./split_image_blur 100 100 1500   (program start step max)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>g
#include <ctype.h>
#include <math.h>

typedef struct {
    int ok;
    int oom;              // detected "Failed to allocate batch memory" etc.
    double wall_ms;
    double imbalance;
    double rec_ratio;     // recommended ratio (0..1), -1 if not found
} RunResult;

static double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static int starts_with(const char* s, const char* pref) {
    return strncmp(s, pref, strlen(pref)) == 0;
}

static int extract_first_double_before_token(const char* line, const char* token, double* out) {
    // Finds first numeric token in line such that next token is `token`
    // Example: "Total wall-clock time: 1262.39 ms" -> 1262.39 with token "ms"
    char buf[1024];
    strncpy(buf, line, sizeof(buf)-1);
    buf[sizeof(buf)-1] = 0;

    // tokenize by whitespace
    char* save = NULL;
    char* prev = NULL;
    for (char* t = strtok_r(buf, " \t\r\n", &save); t; t = strtok_r(NULL, " \t\r\n", &save)) {
        if (prev && strcmp(t, token) == 0) {
            // prev might be number
            char* endp = NULL;
            double v = strtod(prev, &endp);
            if (endp && endp != prev) { *out = v; return 1; }
        }
        prev = t;
    }
    return 0;
}

static int parse_ratio_percent_line(const char* line, double* out_ratio01) {
    // "Recommended GPU ratio: 89.0%" -> 0.89
    const char* key = "Recommended GPU ratio:";
    const char* p = strstr(line, key);
    if (!p) return 0;
    p += strlen(key);
    while (*p && isspace((unsigned char)*p)) p++;

    char num[64];
    int i = 0;
    while (*p && i < (int)sizeof(num)-1) {
        if (isdigit((unsigned char)*p) || *p == '.' ) {
            num[i++] = *p++;
        } else if (*p == '%') {
            break;
        } else {
            // stop on first non-number char (space etc.)
            if (i > 0) break;
            p++;
        }
    }
    num[i] = 0;
    if (i == 0) return 0;

    double pct = atof(num);
    *out_ratio01 = pct / 100.0;
    return 1;
}

static RunResult run_split_image_blur(const char* program, double ratio, int batch) {
    RunResult r;
    r.ok = 0;
    r.oom = 0;
    r.wall_ms = -1.0;
    r.imbalance = -1.0;
    r.rec_ratio = -1.0;

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s %.6f %d", program, ratio, batch);

    FILE* fp = popen(cmd, "r");
    if (!fp) return r;

    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        // detect OOM-like message from your code
        if (strstr(line, "Failed to allocate batch memory") ||
            strstr(line, "Failed to allocate") ||
            strstr(line, "Cannot allocate memory")) {
            r.oom = 1;
        }

        // total time
        if (strstr(line, "Total wall-clock time:")) {
            double v;
            if (extract_first_double_before_token(line, "ms", &v)) r.wall_ms = v;
        }

        // imbalance
        if (strstr(line, "Workload imbalance:")) {
            // parse "...: 5.8%"
            const char* colon = strchr(line, ':');
            if (colon) {
                double v = atof(colon + 1);
                r.imbalance = v;
            }
        }

        // recommended ratio
        {
            double rr;
            if (parse_ratio_percent_line(line, &rr)) r.rec_ratio = rr;
        }
    }

    int status = pclose(fp);
    (void)status;

    // consider ok if we got at least wall_ms and imbalance
    if (r.wall_ms >= 0.0 && r.imbalance >= 0.0 && !r.oom) r.ok = 1;
    return r;
}

static int calibrate_ratio(const char* program, int batch,
                           int calib_iters, double start_ratio,
                           double* out_ratio) {
    // Run calib_iters times, each time read "Recommended GPU ratio"
    // Return average of found recommended ratios (clamped).
    const double RMIN = 0.05, RMAX = 0.95;

    double cur = clamp(start_ratio, RMIN, RMAX);
    double sum = 0.0;
    int got = 0;

    for (int i = 0; i < calib_iters; i++) {
        RunResult rr = run_split_image_blur(program, cur, batch);
        if (rr.oom) return 0;

        if (rr.rec_ratio > 0.0) {
            double rec = clamp(rr.rec_ratio, RMIN, RMAX);
            sum += rec;
            got++;
            cur = rec; // follow the program's suggestion (stabilizes quickly)
        } else {
            // if not printed, still keep current and continue
        }
    }

    if (got == 0) return 0;
    *out_ratio = clamp(sum / (double)got, RMIN, RMAX);
    return 1;
}

static int measure_avg(const char* program, int batch, double ratio,
                       int runs, double* out_ms, double* out_imb) {
    double sum_t = 0.0, sum_i = 0.0;
    int valid = 0;

    // warm-up throwaway
    (void)run_split_image_blur(program, ratio, batch);

    for (int k = 0; k < runs; k++) {
        RunResult rr = run_split_image_blur(program, ratio, batch);
        if (rr.oom) return 0;
        if (rr.ok) {
            sum_t += rr.wall_ms;
            sum_i += rr.imbalance;
            valid++;
        }
    }
    if (valid == 0) return 0;
    *out_ms = sum_t / (double)valid;
    *out_imb = sum_i / (double)valid;
    return 1;
}

typedef struct {
    int batch;
    double ratio;
    double ms;
    double imb;
    int ok;
} Best;

static void try_batch(const char* program, int batch,
                      int calib_iters, int runs,
                      FILE* csv, Best* best_out,
                      double* out_ratio, double* out_ms, double* out_imb) {
    double ratio = 0.5;
    double ms = 0.0, imb = 0.0;

    int ok_cal = calibrate_ratio(program, batch, calib_iters, 0.5, &ratio);
    if (!ok_cal) {
        fprintf(stdout, "  batch=%d: calibration failed\n", batch);
        if (csv) fprintf(csv, "%d,NaN,NaN,NaN,0\n", batch);
        *out_ratio = ratio; *out_ms = NAN; *out_imb = NAN;
        return;
    }

    int ok_meas = measure_avg(program, batch, ratio, runs, &ms, &imb);
    if (!ok_meas) {
        fprintf(stdout, "  batch=%d ratio=%.3f: measurement failed (OOM?)\n", batch, ratio);
        if (csv) fprintf(csv, "%d,%.6f,NaN,NaN,0\n", batch, ratio);
        *out_ratio = ratio; *out_ms = NAN; *out_imb = NAN;
        return;
    }

    fprintf(stdout, "  batch=%d -> ratio=%.3f  avg_time=%.2fms  avg_imb=%.2f%%\n",
            batch, ratio, ms, imb);
    if (csv) fprintf(csv, "%d,%.6f,%.6f,%.6f,1\n", batch, ratio, ms, imb);

    // update best by time (you can also enforce imbalance threshold here if you want)
    if (!best_out->ok || ms < best_out->ms) {
        best_out->ok = 1;
        best_out->batch = batch;
        best_out->ratio = ratio;
        best_out->ms = ms;
        best_out->imb = imb;
    }

    *out_ratio = ratio; *out_ms = ms; *out_imb = imb;
}

int main(int argc, char** argv) {
    const char* program = "./split_image_blur";
    int start_batch = 100;
    int step = 100;
    int max_batch = 1500;

    if (argc >= 2) program = argv[1];
    if (argc >= 3) start_batch = atoi(argv[2]);
    if (argc >= 4) step = atoi(argv[3]);
    if (argc >= 5) max_batch = atoi(argv[4]);

    const int CALIB_ITERS = 3;
    const int RUNS = 3;

    // stopping rule
    const double WORSE_MARGIN = 0.02;   // 2% worse than best counts as worse
    const int WORSE_STREAK_STOP = 2;    // stop after 2 consecutive worse steps

    FILE* csv = fopen("optimization_dynamic.csv", "w");
    if (csv) fprintf(csv, "batch,ratio,avg_time_ms,avg_imbalance,ok\n");

    Best best = {0, 0, 0, 0, 0};

    printf("Dynamic optimizer\n");
    printf("  program: %s\n  start_batch: %d  step: %d  max_batch: %d\n", program, start_batch, step, max_batch);
    printf("  calib_iters: %d  runs_per_measure: %d\n", CALIB_ITERS, RUNS);
    printf("  stop: %d consecutive > %.0f%% worse than best\n\n", WORSE_STREAK_STOP, WORSE_MARGIN*100.0);

    // --- coarse search ---
    int worse_streak = 0;
    for (int batch = start_batch; batch <= max_batch; batch += step) {
        printf("Coarse: testing batch=%d\n", batch);
        double ratio, ms, imb;
        try_batch(program, batch, CALIB_ITERS, RUNS, csv, &best, &ratio, &ms, &imb);

        if (isnan(ms)) {
            // likely OOM or parsing failure; stop increasing
            printf("Stopping coarse search (batch=%d failed)\n\n", batch);
            break;
        }

        if (best.ok && ms > best.ms * (1.0 + WORSE_MARGIN)) worse_streak++;
        else worse_streak = 0;

        if (worse_streak >= WORSE_STREAK_STOP) {
            printf("Stopping coarse search (worse streak reached)\n\n");
            break;
        }
        printf("\n");
    }

    if (!best.ok) {
        printf("No valid configuration found.\n");
        if (csv) fclose(csv);
        return 1;
    }

    // --- refine around best batch (±50, ±25, ±10) ---
    printf("Refine around best batch=%d (time=%.2fms ratio=%.3f)\n\n", best.batch, best.ms, best.ratio);

    int deltas[] = {-50, 50, -25, 25, -10, 10, 0};
    int tried[64];
    int tried_n = 0;

    for (int i = 0; i < (int)(sizeof(deltas)/sizeof(deltas[0])); i++) {
        int b = best.batch + deltas[i];
        if (b <= 0) continue;

        // avoid duplicates
        int dup = 0;
        for (int j = 0; j < tried_n; j++) if (tried[j] == b) { dup = 1; break; }
        if (dup) continue;
        tried[tried_n++] = b;

        printf("Refine: testing batch=%d\n", b);
        double ratio, ms, imb;
        try_batch(program, b, CALIB_ITERS, RUNS, csv, &best, &ratio, &ms, &imb);
        printf("\n");
    }

    printf("====================================\n");
    printf("BEST FOUND:\n");
    printf("  batch=%d\n  ratio=%.3f\n  avg_time=%.2f ms\n  avg_imb=%.2f %%\n",
           best.batch, best.ratio, best.ms, best.imb);
    printf("Run:\n  %s %.6f %d\n", program, best.ratio, best.batch);
    printf("CSV saved: optimization_dynamic.csv\n");
    printf("====================================\n");

    if (csv) fclose(csv);
    return 0;
}
