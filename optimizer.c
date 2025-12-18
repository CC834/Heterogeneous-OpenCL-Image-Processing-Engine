#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_once(double ratio, int batch, double *time_ms, double *imbalance) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "./split_image_blur %.6f %d", ratio, batch);

    FILE *fp = popen(cmd, "r");
    if (!fp) return 0;

    char line[512];
    *time_ms = -1.0;
    *imbalance = -1.0;

    while (fgets(line, sizeof(line), fp)) {
        // Total wall-clock time: 1262.39 ms
        if (strstr(line, "Total wall-clock time:")) {
            double t;
            if (sscanf(line, " %*[^:]: %lf ms", &t) == 1) *time_ms = t;
        }
        // Workload imbalance: 5.8%
        if (strstr(line, "Workload imbalance:")) {
            double im;
            if (sscanf(line, " %*[^:]: %lf%%", &im) == 1) *imbalance = im;
        }
    }

    pclose(fp);
    return (*time_ms >= 0.0 && *imbalance >= 0.0);
}

int main() {
    int batches[] = {50, 100, 250, 500, 650, 1000};
    double ratios[] = {0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95};

    double best_t = 1e18, best_r = 0.0, best_im = 1e18;
    int best_b = 0;

    for (int bi=0; bi<6; bi++) {
        int batch = batches[bi];
        printf("=== batch=%d ===\n", batch);

        for (int ri=0; ri<11; ri++) {
            double r = ratios[ri], t, im;
            if (!run_once(r, batch, &t, &im)) {
                printf("  r=%.2f parse fail\n", r);
                continue;
            }
            printf("  r=%.2f  time=%.2fms  imb=%.2f%%\n", r, t, im);

            // Example: choose best time (you can enforce imbalance here)
            if (t < best_t) { best_t=t; best_r=r; best_b=batch; best_im=im; }
        }
    }

    printf("\nBEST: batch=%d ratio=%.3f time=%.2fms imb=%.2f%%\n", best_b, best_r, best_t, best_im);
    return 0;
}
