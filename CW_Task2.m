% Newton–Raphson PF for Task 2: sweep SOP active power and plot voltages
clear; clc; close all; format long;

% ---------------- Data ----------------
V1 = 1.01; theta1 = 0;                    % slack
Z12 = 0.04 + 1i*0;  y12 = 1/Z12;
Z23 = 0.02 + 1i*0;  y23 = 1/Z23;
Z14 = 0.04 + 1i*0;  y14 = 1/Z14;

G = [50,-25,0,-25; -25,75,-50,0; 0,-50,50,0; -25,0,0,25];
N = 4;

% Base net complex injections S = P + jQ (generation +, load -)
Sbase = zeros(N,1);
Sbase(2) = +0.85;
Sbase(3) = -0.01;
Sbase(4) = -0.50 - 1i*0.10;

% ---------------- SOP sweep ----------------
PSOP_vec = linspace(-1, 1, 101);         % sweep from -1 to +1 pu
nm = numel(PSOP_vec);

Vsave = nan(N, nm);                     % store complex bus voltages
its   = nan(1, nm);                     % iterations used
conv  = false(1, nm);                   % convergence flag

% Initial guess (warm-start across sweep)
theta0 = zeros(N,1); theta0(1) = theta1;
Vmag0  = ones(N,1);  Vmag0(1)  = V1;

tol = 1e-10; maxIt = 50;
PQ = [2 3 4]; npq = numel(PQ);

for m = 1:nm
    PSOP = PSOP_vec(m);

    % Compose injections for this SOP (QSOP=0 at both sides)
    S = Sbase;
    S(3) = complex(real(S(3)) - PSOP, imag(S(3)) + 0.0);  % +PSOP at bus 3
    S(4) = complex(real(S(4)) + PSOP, imag(S(4)) + 0.0);  % -PSOP at bus 4

    % ---------------- NR iterations (same core as your code) ----------------
    theta = theta0; Vmag = Vmag0;       % warm-start from previous case

    for it = 1:maxIt
        % Calculate P and Q injections from network model (B=0)
        Pcalc = zeros(N,1); Qcalc = zeros(N,1);
        for i = 1:N
            for k = 1:N
                dth = theta(i) - theta(k);
                Pcalc(i) = Pcalc(i) + Vmag(i)*Vmag(k)*G(i,k)*cos(dth);
                Qcalc(i) = Qcalc(i) + Vmag(i)*Vmag(k)*G(i,k)*sin(dth);
            end
        end

        % Mismatch (only for PQ buses)
        Pspec = real(S);  Qspec = imag(S);
        dP = Pspec(PQ) - Pcalc(PQ);
        dQ = Qspec(PQ) - Qcalc(PQ);
        mis = [dP; dQ];

        if max(abs(mis)) < tol
            conv(m) = true; its(m) = it; break;
        end

        % Jacobian blocks H, N, M, L
        H = zeros(npq); Nn = zeros(npq);
        M = zeros(npq); L = zeros(npq);

        for a = 1:npq
            i = PQ(a);
            % Diagonals
            H(a,a) = -Qcalc(i);                         % -Q_i
            M(a,a) =  Pcalc(i) - Vmag(i)^2 * G(i,i);    % P_i - V_i^2*G_ii

            sumN = 0; sumL = 0;
            for k = 1:N
                if k == i, continue; end
                dth = theta(i) - theta(k);
                sumN = sumN + Vmag(k)*G(i,k)*cos(dth);
                sumL = sumL + Vmag(k)*G(i,k)*sin(dth);
            end
            Nn(a,a) = 2*Vmag(i)*G(i,i) + sumN;          % ∂P_i/∂|V_i|
            L(a,a)  = sumL;                              % ∂Q_i/∂|V_i|

            % Off-diagonals
            for b = 1:npq
                j = PQ(b); if j == i, continue; end
                dth = theta(i) - theta(j);
                H(a,b) =  Vmag(i)*Vmag(j)*G(i,j)*sin(dth);
                M(a,b) = -Vmag(i)*Vmag(j)*G(i,j)*cos(dth);
                Nn(a,b) =  Vmag(i)*G(i,j)*cos(dth);
                L(a,b)  =  Vmag(i)*G(i,j)*sin(dth);
            end
        end

        % Solve for updates [Δθ_PQ; Δ|V|_PQ]
        J = [H, Nn; M, L];
        dx = J \ mis;

        % Update states
        theta(PQ) = theta(PQ) + dx(1:npq);
        Vmag(PQ)  = Vmag(PQ)  + dx(npq+1:end);
    end

    % Save solution
    V = Vmag.* exp(1i*theta);
    Vsave(:,m) = V;
    theta0 = theta; Vmag0 = Vmag;
end

% ---------------- Plot voltages vs PSOP ----------------
figure('Color','w'); hold on; grid on; box on;
cols = lines(4);
for i = 1:4
    plot(PSOP_vec(conv), abs(Vsave(i,conv)), 'LineWidth', 1.8, 'Color', cols(i,:));
end
yline(0.96,'k--','0.96 pu'); yline(1.04,'k--','1.04 pu');
xlabel('P_{SOP} (pu)');
ylabel('|V| (pu)');
legend('Bus 1','Bus 2','Bus 3','Bus 4','Location','best');
title('Bus voltages vs SOP active-power setpoint');

% ---------------- Report PSOP ranges keeping |V3| within [0.96, 1.04] pu ----------------
stat = [0.96, 1.04];
ok3 = conv & abs(Vsave(3,:)) >= stat(1) & abs(Vsave(3,:)) <= stat(2);
ranges = contiguous_ranges(PSOP_vec, ok3);

fprintf('\nConverged points: %d/%d. Average NR iterations: %.2f\n', sum(conv), nm, mean(its(conv)));
fprintf('PSOP ranges that keep |V3| within [%.2f, %.2f] pu:\n', stat(1), stat(2));
if isempty(ranges)
    fprintf('  None within the scanned setpoints.\n');
else
    for r = 1:size(ranges,1)
        fprintf('  [%.4f, %.4f] pu\n', ranges(r,1), ranges(r,2));
    end
end

% ===== Utility: contiguous ranges from a logical mask =====
function R = contiguous_ranges(x, mask)
    R = [];
    if ~any(mask), return; end
    d = diff([false, mask, false]);
    i1 = find(d==1); i2 = find(d==-1)-1;
    R = [x(i1).' x(i2).'];
end
