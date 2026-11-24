// Classical solver for Eqs. 14 to 19: Keller box scheme, Newton, banded LU, Richardson.
//
// Build:  g++ -O2 -std=c++17 -o engine/bvp_engine engine/bvp_engine.cpp
// Run:    engine/bvp_engine [case.json] [out_dir] [L]
//         engine/bvp_engine --estimate <pinn.csv> <out.csv>   (one Newton step from a PINN state)
//         engine/bvp_engine --deflate [L]                     (branch search, engine/out/engine_branches.csv)
//         defaults: configs/physical-case-pending.json  engine/out  20
//
// Method (the standard scheme for boundary-layer ODEs; Keller 1970s, see engine/README.md):
//   1. Eqs. 14 to 18 become 12 first-order ODEs z' = F(eta, z), the same rewrite as
//      reference/equations.py (rhs): solve M*H = -r0 for H = [f4, g2, th2, ph2, ch2].
//   2. On each mesh interval: (z[j+1] - z[j]) / h = F(eta_mid, (z[j] + z[j+1]) / 2).
//      Second-order accurate. With B1 to B12 this is 12*N equations for 12*N unknowns.
//   3. Newton's method. The Jacobian is banded (17 below, 17 above the diagonal) because
//      the boundary conditions are separated, so a banded LU solves each step in O(N).
//   4. The mesh is halved four times. Two neighbouring levels give a Richardson value
//      (4*fine - coarse)/3, fourth order, and the observed order of the scheme.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using State = std::array<double, 12>;  // f f1 f2 f3 g g1 theta th1 phi ph1 chi ch1
using High = std::array<double, 5>;    // f4 g2 th2 ph2 ch2

struct Params {
  double Delta, B, Da, kappa, delta_star, theta_w, lambda_m, beta_m, epsilon, b, Pr, Le, N1, Bi, Rd,
      Nr, Nt, Nb, Lb, Pe, ell, fw;
};

// Read the 22 values from the "physical_parameters" block of the locked case file.
Params read_case(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string s = ss.str();
  size_t a = s.find("\"physical_parameters\"");
  size_t e = s.find('}', a);
  if (a == std::string::npos || e == std::string::npos) throw std::runtime_error("no physical_parameters");
  std::string blk = s.substr(a, e - a);
  auto get = [&](const char* name) {
    std::string key = std::string("\"") + name + "\":";
    size_t k = blk.find(key);
    if (k == std::string::npos) throw std::runtime_error(std::string("missing ") + name);
    return std::strtod(blk.c_str() + k + key.size(), nullptr);
  };
  return {get("Delta"), get("B"),  get("Da"), get("kappa"), get("delta_star"), get("theta_w"),
          get("lambda_m"), get("beta_m"), get("epsilon"), get("b"), get("Pr"), get("Le"),
          get("N1"), get("Bi"), get("Rd"), get("Nr"), get("Nt"), get("Nb"), get("Lb"), get("Pe"),
          get("ell"), get("fw")};
}

// R14 to R18 exactly as in docs/problem/equations.md. With H = 0 this is r0.
High residuals(double eta, const State& z, const High& H, const Params& p) {
  auto [f, f1, f2, f3, g, g1, th, th1, ph, ph1, ch, ch1] = z;
  auto [f4, g2, th2, ph2, ch2] = H;
  double k = p.kappa, D = p.Delta, ds = p.delta_star, Pr = p.Pr, q = eta + k, h = eta + p.b;
  double c = p.theta_w - 1.0, A = 1.0 + c * th;
  double R14 = (1 + D) * (f4 + 2 * f3 / q - f2 / (q * q) + f1 / (q * q * q)) + k / q * (f * f3 - f1 * f2) +
               k / (q * q) * (f * f2 - f1 * f1) - k / (q * q * q) * f * f1 -
               0.5 * ds * ((eta * f3 + 3 * f2) + (0.5 * eta * f2 + f1) / q) - (f2 + f1 / q) / p.Da -
               D * (g2 + g1 / q) - 2 * p.beta_m / std::pow(h, 4) * ((1 / q - 4 / h) * th + th1);
  double R15 = (1 + D / 2) * (g2 + g1 / q) - D * p.B * (f2 + f1 / q + 2 * g) - k / q * (f1 * g - f * g1) -
               ds / 2 * (eta * g1 + 3 * g);
  double R16 = (th2 + th1 / q) / Pr + p.Nr / Pr * (A * A * A * th1 / q + 3 * c * A * A * th1 * th1 + A * A * A * th2) +
               k / q * f * th1 + p.Nt / Pr * th1 * th1 + p.Nb / Pr * th1 * ph1 +
               2 * p.beta_m * p.lambda_m * (th - p.epsilon) / (Pr * h * h * h) * (k * f / q * (1 - 2 / (h * h)) - 2 * f1 / h) -
               ds * eta / 2 * th1;
  double R17 = (ph2 + ph1 / q) / (p.Le * Pr) + p.Nt / (p.Le * Pr * p.Nb) * (th2 + th1 / q) + k / q * f * ph1 -
               ds * eta / 2 * ph1;
  double R18 = (ch2 + ch1 / q) / p.Lb + k / q * f * ph + p.Pe / Pr * (ch * ph2 + ch1 * ph1) - ds * eta / 2 * ph1;
  return {R14, R15, R16, R17, R18};
}

// Solve M*H = -r0. M is triangular apart from one entry, so this is back substitution.
High highest(double eta, const State& z, const Params& p) {
  High r = residuals(eta, z, {0, 0, 0, 0, 0}, p), H;
  double A = 1.0 + (p.theta_w - 1.0) * z[6], D = p.Delta;
  H[1] = -r[1] / (1 + D / 2);                                       // g2
  H[0] = (-r[0] + D * H[1]) / (1 + D);                              // f4
  H[2] = -r[2] / ((1 + p.Nr * A * A * A) / p.Pr);                   // th2
  H[3] = (-r[3] - p.Nt / (p.Le * p.Pr * p.Nb) * H[2]) * p.Le * p.Pr;  // ph2
  H[4] = (-r[4] - p.Pe * z[10] / p.Pr * H[3]) * p.Lb;              // ch2
  return H;
}

State rhs(double eta, const State& z, const Params& p) {
  High H = highest(eta, z, p);
  return {z[1], z[2], z[3], H[0], z[5], H[1], z[7], H[2], z[9], H[3], z[11], H[4]};
}

// B1 to B6 at eta = 0 (stable form of B4); B7 to B12 at eta = L, with the asymptotic
// far conditions for theta and phi (equations.far_conditions).
std::array<double, 6> wall_bc(const State& a, const Params& p) {
  double Aw = 1 + (p.theta_w - 1) * a[6];
  double Qw = (1 - a[6]) * (p.theta_w + Aw) * (p.theta_w * p.theta_w + Aw * Aw);
  return {a[1] - 1 - p.N1 * (a[2] - a[1] / p.kappa), a[0] - p.fw / (p.Pr * p.Le) * a[9], a[4] + p.ell * a[2],
          -a[7] - p.Bi * (1 - a[6]) - p.Rd * Qw, a[9] + p.Nt / p.Nb * a[7], a[10] - 1};
}

std::array<double, 6> far_bc(const State& b, double L, const Params& p) {
  double q = L + p.kappa, A = 1 + (p.theta_w - 1) * b[6];
  double m = p.Pr * p.kappa * b[0] / (1 + p.Nr * A * A * A), mp = p.Le * p.Pr * p.kappa * b[0];
  return {b[1], b[2], b[4], b[7] + m * b[6] / q, b[9] + (mp * b[8] - p.Nt / p.Nb * m * b[6]) / q, b[10]};
}

// Banded matrix with partial pivoting. Row r stores columns r-kl .. r+kl+ku (room for fill-in).
struct Band {
  int n, kl, ku, w;
  std::vector<double> a;
  Band(int n_, int kl_, int ku_) : n(n_), kl(kl_), ku(ku_), w(2 * kl_ + ku_ + 1), a((size_t)n_ * w, 0.0) {}
  double& at(int r, int c) { return a[(size_t)r * w + (c - r + kl)]; }
  void solve(std::vector<double>& b) {  // overwrites the matrix; b becomes the solution
    for (int k = 0; k < n; ++k) {
      int last = std::min(n - 1, k + kl), right = std::min(n - 1, k + kl + ku), piv = k;
      for (int i = k + 1; i <= last; ++i)
        if (std::fabs(at(i, k)) > std::fabs(at(piv, k))) piv = i;
      if (at(piv, k) == 0.0) throw std::runtime_error("singular Jacobian");
      if (piv != k) {
        for (int c = k; c <= right; ++c) std::swap(at(k, c), at(piv, c));
        std::swap(b[k], b[piv]);
      }
      for (int i = k + 1; i <= last; ++i) {
        double m = at(i, k) / at(k, k);
        if (m == 0.0) continue;
        for (int c = k + 1; c <= right; ++c) at(i, c) -= m * at(k, c);
        b[i] -= m * b[k];
      }
    }
    for (int k = n - 1; k >= 0; --k) {
      double s = b[k];
      for (int c = k + 1; c <= std::min(n - 1, k + kl + ku); ++c) s -= at(k, c) * b[c];
      b[k] = s / at(k, k);
    }
  }
};

// Residual of the whole discrete problem. Order: 6 wall conditions, 12 box equations per
// interval, 6 far conditions. That order keeps the Jacobian banded.
std::vector<double> discrete_residual(const std::vector<double>& x, const std::vector<State>& z, const Params& p) {
  int N = (int)x.size();
  std::vector<double> R(12 * N);
  auto wb = wall_bc(z[0], p);
  for (int i = 0; i < 6; ++i) R[i] = wb[i];
  for (int j = 0; j + 1 < N; ++j) {
    double h = x[j + 1] - x[j];
    State mid;
    for (int i = 0; i < 12; ++i) mid[i] = 0.5 * (z[j][i] + z[j + 1][i]);
    State F = rhs(0.5 * (x[j] + x[j + 1]), mid, p);
    for (int i = 0; i < 12; ++i) R[6 + 12 * j + i] = (z[j + 1][i] - z[j][i]) / h - F[i];
  }
  auto fb = far_bc(z[N - 1], x[N - 1], p);
  for (int i = 0; i < 6; ++i) R[12 * N - 6 + i] = fb[i];
  return R;
}

double max_abs(const std::vector<double>& v) {
  double m = 0;
  for (double a : v) m = std::max(m, std::fabs(a));
  return m;
}

// Jacobian of discrete_residual at z. The Jacobian of F comes from forward differences (step 1e-7 relative).
Band jacobian(const std::vector<double>& x, const std::vector<State>& z, const Params& p) {
  int N = (int)x.size(), n = 12 * N;
  Band J(n, 17, 17);
  auto put_bc = [&](int row0, int col0, auto bcfun, const State& s) {
    auto b0 = bcfun(s);
    for (int c = 0; c < 12; ++c) {
      State t = s;
      double d = 1e-7 * std::max(1.0, std::fabs(s[c]));
      t[c] += d;
      auto b1 = bcfun(t);
      for (int r = 0; r < 6; ++r) J.at(row0 + r, col0 + c) = (b1[r] - b0[r]) / d;
    }
  };
  put_bc(0, 0, [&](const State& s) { return wall_bc(s, p); }, z[0]);
  for (int j = 0; j + 1 < N; ++j) {
    double h = x[j + 1] - x[j], em = 0.5 * (x[j] + x[j + 1]);
    State mid;
    for (int i = 0; i < 12; ++i) mid[i] = 0.5 * (z[j][i] + z[j + 1][i]);
    State F0 = rhs(em, mid, p);
    for (int c = 0; c < 12; ++c) {
      State t = mid;
      double d = 1e-7 * std::max(1.0, std::fabs(mid[c]));
      t[c] += d;
      State F1 = rhs(em, t, p);
      for (int r = 0; r < 12; ++r) {
        double dF = 0.5 * (F1[r] - F0[r]) / d;  // d(mid)/d(z_j) = d(mid)/d(z_j+1) = 1/2
        J.at(6 + 12 * j + r, 12 * j + c) = (r == c ? -1.0 / h : 0.0) - dF;
        J.at(6 + 12 * j + r, 12 * (j + 1) + c) = (r == c ? 1.0 / h : 0.0) - dF;
      }
    }
  }
  put_bc(n - 6, n - 12, [&](const State& s) { return far_bc(s, x[N - 1], p); }, z[N - 1]);
  return J;
}

// Newton with step halving.
int newton(const std::vector<double>& x, std::vector<State>& z, const Params& p) {
  int N = (int)x.size();
  auto R = discrete_residual(x, z, p);
  for (int it = 0; it < 50; ++it) {
    if (max_abs(R) < 1e-11) return it;
    Band J = jacobian(x, z, p);
    std::vector<double> step = R;
    J.solve(step);
    if (max_abs(step) < 1e-12) return it + 1;  // Newton step at rounding level: converged
    double lam = 1.0, r0 = max_abs(R);
    for (int k = 0; k < 30; ++k, lam /= 2) {
      std::vector<State> trial = z;
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < 12; ++i) trial[j][i] -= lam * step[12 * j + i];
      auto Rt = discrete_residual(x, trial, p);
      if (std::isfinite(max_abs(Rt)) && max_abs(Rt) < r0) {
        z = trial;
        R = Rt;
        break;
      }
      if (k == 29) {
        if (r0 < 1e-8) return it + 1;  // the residual is at its rounding floor and cannot fall further
        throw std::runtime_error("Newton line search failed");
      }
    }
    if (lam * max_abs(step) < 1e-13) return it + 1;
  }
  throw std::runtime_error("Newton did not converge in 50 steps");
}

// n points on [0, L], exponentially stretched so the first step is first_step (as solver.wall_mesh).
std::vector<double> wall_mesh(double L, int n, double first_step) {
  auto x_of = [&](double beta, double s) { return L * std::expm1(beta * s) / std::expm1(beta); };
  double lo = 1e-9, hi = 500.0;
  for (int k = 0; k < 200; ++k) {
    double mid = 0.5 * (lo + hi);
    (x_of(mid, 1.0 / (n - 1)) > first_step ? lo : hi) = mid;
  }
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) x[i] = x_of(0.5 * (lo + hi), double(i) / (n - 1));
  x[0] = 0.0;
  x[n - 1] = L;
  return x;
}

// Smooth decaying guess that meets B1, B2, B3, B5, B6 (as solver.smooth_guess).
State guess(double eta, const Params& p, double rate, double theta0) {
  double e = std::exp(-rate * eta), a = 1.0 / (1.0 + p.N1 * rate + p.N1 / p.kappa), ratio = p.Nt / p.Nb;
  double f0 = p.fw / (p.Pr * p.Le) * ratio * theta0 * rate, th = theta0 * e, g = p.ell * a * rate * e;
  return {f0 + a / rate * (1 - e), a * e, -a * rate * e, a * rate * rate * e, g, -rate * g,
          th, -rate * th, -ratio * th, ratio * rate * th, e, -rate * e};
}

void write_csv(const std::string& path, const std::vector<double>& x, const std::vector<State>& z, const Params& p) {
  std::ofstream out(path);
  out << "eta,f,f1,f2,f3,g,g1,theta,th1,phi,ph1,chi,ch1,f4,g2,th2,ph2,ch2\n";
  out.precision(15);
  for (size_t j = 0; j < x.size(); ++j) {
    High H = highest(x[j], z[j], p);
    out << x[j];
    for (double v : z[j]) out << ',' << v;
    for (double v : H) out << ',' << v;
    out << '\n';
  }
}

// Error estimate for a PINN (gap R6): one Newton step of the box scheme from the PINN state.
// For a state z near the solution z*, the step J^-1 R(z) equals z - z* up to second-order terms
// and the box scheme's own error (GCI about 1e-7, reference/error_bars.md). It needs no
// reference solution, so it can be reported for any run. Input: CSV with a header and columns
// eta, then the 12 states, on any increasing mesh from 0 to L. Output: eta and the 12 estimated
// errors (PINN minus solution).
int estimate(const std::string& in_path, const std::string& out_path, const Params& p) {
  std::ifstream in(in_path);
  if (!in) throw std::runtime_error("cannot read " + in_path);
  std::string line;
  std::getline(in, line);
  std::vector<double> x;
  std::vector<State> z;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> v;
    while (std::getline(ss, cell, ',')) v.push_back(std::stod(cell));
    if (v.size() < 13) throw std::runtime_error("each row needs eta and 12 states");
    x.push_back(v[0]);
    State s;
    for (int i = 0; i < 12; ++i) s[i] = v[1 + i];
    z.push_back(s);
  }
  auto R = discrete_residual(x, z, p);
  Band J = jacobian(x, z, p);
  std::vector<double> step = R;
  J.solve(step);
  std::ofstream out(out_path);
  out.precision(12);
  out << "eta,f,f1,f2,f3,g,g1,theta,th1,phi,ph1,chi,ch1\n";
  for (size_t j = 0; j < x.size(); ++j) {
    out << x[j];
    for (int i = 0; i < 12; ++i) out << ',' << step[12 * j + i];
    out << '\n';
  }
  std::printf("discrete residual max %.3e on %zu nodes; estimated wall errors: f''(0) %.3e, theta(0) %.3e, "
              "theta'(0) %.3e, phi'(0) %.3e, chi'(0) %.3e\n", max_abs(R), x.size(), step[2], step[6], step[7],
              step[9], step[11]);
  return 0;
}

// Branch search by deflation (reference audit D7). Deflation: Farrell, Birkisson and Funke,
// "Deflation techniques for finding distinct solutions of nonlinear PDEs", SIAM J. Sci. Comput. 37
// (2015). Newton is run on G(z) = M(z) R(z) with M(z) = prod_i (|z - r_i|^-2 + 1) over the roots
// r_i found so far. G has the same roots as R except the known ones, so from the same start Newton
// either finds a new solution of the locked problem or fails. No parameter value is changed.
// With d the plain Newton direction, the deflated direction is d / (1 - (grad M . d) / M).
double rms(const std::vector<State>& a, const std::vector<State>* b = nullptr) {
  double s = 0;
  for (size_t j = 0; j < a.size(); ++j)
    for (int i = 0; i < 12; ++i) {
      double e = a[j][i] - (b ? (*b)[j][i] : 0.0);
      s += e * e;
    }
  return std::sqrt(s / (12.0 * a.size()));
}

// Newton on the deflated residual. Returns the step count, or -1 if it fails to converge.
int deflated_newton(const std::vector<double>& x, std::vector<State>& z, const Params& p,
                    const std::vector<std::vector<State>>& roots) {
  int N = (int)x.size();
  double n = 12.0 * N;
  auto deflation = [&](const std::vector<State>& s) {
    double M = 1;
    for (const auto& r : roots) M *= std::pow(rms(s, &r), -2) + 1.0;
    return M;
  };
  try {
    auto R = discrete_residual(x, z, p);
    for (int it = 0; it < 100; ++it) {
      if (max_abs(R) < 1e-10) return it;
      Band J = jacobian(x, z, p);
      std::vector<double> step = R;  // z_new = z - tau * lam * step
      J.solve(step);
      double ratio = 0;  // (grad M . d) / M with d = -step
      for (const auto& r : roots) {
        double e = rms(z, &r), dot = 0;
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < 12; ++i) dot += (z[j][i] - r[j][i]) * (-step[12 * j + i]);
        ratio += (-2 * std::pow(e, -4) * dot / n) / (std::pow(e, -2) + 1.0);
      }
      double tau = 1.0 / (1.0 - ratio);
      if (!std::isfinite(tau)) return -1;
      double merit = max_abs(R) * deflation(z), lam = 1.0;
      bool moved = false;
      for (int k = 0; k < 30; ++k, lam /= 2) {
        std::vector<State> trial = z;
        for (int j = 0; j < N; ++j)
          for (int i = 0; i < 12; ++i) trial[j][i] -= tau * lam * step[12 * j + i];
        auto Rt = discrete_residual(x, trial, p);
        double mt = max_abs(Rt) * deflation(trial);
        if (std::isfinite(mt) && mt < merit) {
          z = trial;
          R = Rt;
          moved = true;
          break;
        }
      }
      if (!moved) return -1;
    }
  } catch (const std::exception&) {
    return -1;
  }
  return -1;
}

// Halve every interval and run plain Newton, to check that a root found on the coarse mesh
// persists on the fine one (a discretization artifact would not).
bool confirm(std::vector<double> x, std::vector<State> z, const Params& p, State& wall) {
  std::vector<double> xf;
  std::vector<State> zf;
  for (size_t j = 0; j + 1 < x.size(); ++j) {
    xf.push_back(x[j]);
    zf.push_back(z[j]);
    xf.push_back(0.5 * (x[j] + x[j + 1]));
    State m;
    for (int i = 0; i < 12; ++i) m[i] = 0.5 * (z[j][i] + z[j + 1][i]);
    zf.push_back(m);
  }
  xf.push_back(x.back());
  zf.push_back(z.back());
  try {
    newton(xf, zf, p);
  } catch (const std::exception& e) {
    std::printf("  refinement: %s\n", e.what());
    return false;
  }
  wall = zf[0];
  double change = 0;  // largest relative change of a wall state, scale at least 1
  int worst = 0;
  for (int i = 0; i < 12; ++i) {
    double c = std::fabs(zf[0][i] - z[0][i]) / std::max(1.0, std::fabs(z[0][i]));
    if (c > change) change = c, worst = i;
  }
  std::printf("  refinement: largest relative wall change %.2e (state %d)\n", change, worst);
  return change < 1e-2;
}

int deflate(double L, const std::string& out_path, const Params& p) {
  std::vector<double> x = wall_mesh(L, 401, 2e-3);
  std::vector<std::vector<State>> roots;
  std::ofstream out(out_path);
  out.precision(12);
  out << "root,start_rate,start_theta0,deflated_against,newton_steps,f2_0,theta_0,th1_0,phi_0,ph1_0,g1_0,ch1_0,f_L,"
         "m_far,confirmed_801_nodes,f2_0_801\n";
  int tried = 0;
  for (double rate : {0.5, 1.0, 2.0, 4.0}) {
    for (int t = 0; t <= 12; ++t) {
      double theta0 = -0.5 + 0.25 * t;
      for (int pass = 0; pass < 4; ++pass) {  // plain Newton first, then deflated against the roots so far
        std::vector<State> z(x.size());
        for (size_t j = 0; j < x.size(); ++j) z[j] = guess(x[j], p, rate, theta0);
        ++tried;
        int known = (int)roots.size();
        int steps = deflated_newton(x, z, p, pass == 0 ? std::vector<std::vector<State>>{} : roots);
        if (steps < 0) break;
        bool is_new = true;
        for (const auto& r : roots)
          if (rms(z, &r) < 1e-6 * std::max(1.0, rms(r))) is_new = false;
        if (!is_new) {
          if (pass == 0) continue;  // plain Newton found a known root: now deflate it
          break;
        }
        roots.push_back(z);
        State wall{};
        bool ok = confirm(x, z, p, wall);
        double A = 1 + (p.theta_w - 1) * z.back()[6];
        double m = p.Pr * p.kappa * z.back()[0] / (1 + p.Nr * A * A * A);
        out << roots.size() << ',' << rate << ',' << theta0 << ',' << known << ',' << steps << ',' << z[0][2] << ','
            << z[0][6] << ',' << z[0][7] << ',' << z[0][8] << ',' << z[0][9] << ',' << z[0][5] << ',' << z[0][11]
            << ',' << z.back()[0] << ',' << m << ',' << (ok ? 1 : 0) << ',' << (ok ? wall[2] : NAN) << '\n';
        std::printf("root %zu from rate %.2f theta0 %.2f (deflated against %d): f''(0) %.6f theta(0) %.6f f(L) %.6f "
                    "m %.4f, %s on 801 nodes\n", roots.size(), rate, theta0, known, z[0][2], z[0][6], z.back()[0], m,
                    ok ? "confirmed" : "NOT confirmed");
      }
    }
  }
  std::printf("%d Newton runs from %d starts; %zu distinct solutions; wrote %s\n", tried, 52, roots.size(),
              out_path.c_str());
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--deflate") {
    try {
      Params p = read_case(argc > 3 ? argv[3] : "configs/physical-case-pending.json");
      std::filesystem::create_directories("engine/out");
      return deflate(argc > 2 ? std::atof(argv[2]) : 20.0, "engine/out/engine_branches.csv", p);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc > 1 && std::string(argv[1]) == "--estimate") {
    if (argc < 4) {
      std::fprintf(stderr, "usage: bvp_engine --estimate <pinn.csv> <out.csv> [case.json]\n");
      return 2;
    }
    try {
      return estimate(argv[2], argv[3], read_case(argc > 4 ? argv[4] : "configs/physical-case-pending.json"));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  std::string case_path = argc > 1 ? argv[1] : "configs/physical-case-pending.json";
  std::string out_dir = argc > 2 ? argv[2] : "engine/out";
  double L = argc > 3 ? std::atof(argv[3]) : 20.0;
  try {
    Params p = read_case(case_path);
    std::filesystem::create_directories(out_dir);
    // Level 0 from the smooth guess (rate 1, theta(0) 0.3 reaches branch A in the reference scan).
    std::vector<double> x = wall_mesh(L, 401, 2e-3);
    std::vector<State> z(x.size());
    for (size_t j = 0; j < x.size(); ++j) z[j] = guess(x[j], p, 1.0, 0.3);
    std::vector<std::vector<double>> xs;
    std::vector<std::vector<State>> zs;
    std::ofstream conv(out_dir + "/engine_convergence.csv");
    conv << "level,nodes,newton_steps,seconds,f2_0,th1_0,theta_0,phi_0,ph1_0,g1_0,ch1_0,f_L,max_change\n";
    conv.precision(12);
    std::printf("level  nodes  newton  seconds   f''(0)          theta(0)        max change\n");
    for (int level = 0; level < 5; ++level) {
      if (level > 0) {  // halve every interval; start Newton from the averaged coarse solution
        std::vector<double> xf;
        std::vector<State> zf;
        for (size_t j = 0; j + 1 < x.size(); ++j) {
          xf.push_back(x[j]);
          zf.push_back(z[j]);
          xf.push_back(0.5 * (x[j] + x[j + 1]));
          State m;
          for (int i = 0; i < 12; ++i) m[i] = 0.5 * (z[j][i] + z[j + 1][i]);
          zf.push_back(m);
        }
        xf.push_back(x.back());
        zf.push_back(z.back());
        x = xf;
        z = zf;
      }
      auto t0 = std::chrono::steady_clock::now();
      int steps = newton(x, z, p);
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double change = NAN;
      if (level > 0) {  // compare on the coarse nodes, which are every second fine node
        change = 0;
        for (size_t j = 0; j < zs.back().size(); ++j)
          for (int i = 0; i < 12; ++i) change = std::max(change, std::fabs(z[2 * j][i] - zs.back()[j][i]));
      }
      conv << level << ',' << x.size() << ',' << steps << ',' << sec << ',' << z[0][2] << ',' << z[0][7] << ','
           << z[0][6] << ',' << z[0][8] << ',' << z[0][9] << ',' << z[0][5] << ',' << z[0][11] << ','
           << z.back()[0] << ',' << change << '\n';
      std::printf("%5d %6zu %7d %8.3f  %.12f %.12f %.2e\n", level, x.size(), steps, sec, z[0][2], z[0][6], change);
      xs.push_back(x);
      zs.push_back(z);
    }
    // Observed order from the last three levels, and the Richardson value on the second-finest mesh.
    int K = (int)zs.size() - 1;
    double d1 = 0, d2 = 0;
    for (size_t j = 0; j < zs[K - 2].size(); ++j)
      for (int i = 0; i < 12; ++i) {
        d1 = std::max(d1, std::fabs(zs[K - 1][2 * j][i] - zs[K - 2][j][i]));
        d2 = std::max(d2, std::fabs(zs[K][4 * j][i] - zs[K - 1][2 * j][i]));
      }
    std::printf("observed order of the box scheme: %.2f (expected 2)\n", std::log2(d1 / d2));
    std::vector<State> rich(zs[K - 1].size());
    for (size_t j = 0; j < rich.size(); ++j)
      for (int i = 0; i < 12; ++i) rich[j][i] = (4 * zs[K][2 * j][i] - zs[K - 1][j][i]) / 3;
    write_csv(out_dir + "/engine_box_finest.csv", xs[K], zs[K], p);
    write_csv(out_dir + "/engine_solution.csv", xs[K - 1], rich, p);
    std::printf("Richardson f''(0) = %.10f, theta(0) = %.10f, theta'(0) = %.10f\n", rich[0][2], rich[0][6], rich[0][7]);
    std::printf("wrote %s/engine_solution.csv (Richardson, %zu nodes), engine_box_finest.csv, engine_convergence.csv\n",
                out_dir.c_str(), rich.size());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
