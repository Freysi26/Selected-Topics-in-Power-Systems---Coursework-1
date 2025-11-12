% Newton–Raphson power flow for the 4-bus feeder (PQ buses: 2,3,4; bus 1 slack)
clear; clc; format long;

% ---------------- Data ----------------
V1 = 1.01; theta1 = 0;                    % slack
Z12 = 0.04 + 1i*0;  y12 = 1/Z12;
Z23 = 0.02 + 1i*0;  y23 = 1/Z23;
Z14 = 0.04 + 1i*0;  y14 = 1/Z14;

G = [50,-25,0,-25;-25,75,-50,0;0,-50,50,0;-25,0,0,25];

% Net complex injections S = P + jQ at buses (generation +, load -)
N = 4;
SOP = 0.4;
S = zeros(N,1);
S(2) = +0.85 -0.85;
S(3) = -0.01 -SOP;
S(4) = -0.50 - 1i*0.10 +SOP;

% ---------------- Unknowns and settings ----------------
PQ = [2 3 4]; npq = numel(PQ);
theta = zeros(N,1); theta(1) = theta1;
Vmag  = ones(N,1);  Vmag(1)  = V1;

tol = 1e-10; maxIt = 50;

% ---------------- NR iterations ----------------
for it = 1:maxIt
    % Calculate P and Q injections from network model
    Pcalc = zeros(N,1); Qcalc = zeros(N,1);
    for i = 1:N
        for k = 1:N
            dth = theta(i) - theta(k);
            Pcalc(i) = Pcalc(i) + Vmag(i)*Vmag(k)*( G(i,k)*cos(dth));
            Qcalc(i) = Qcalc(i) + Vmag(i)*Vmag(k)*( G(i,k)*sin(dth));
        end
    end

    % Mismatch (only for PQ buses)
    Pspec = real(S);  Qspec = imag(S);      % injections as specified
    dP = Pspec(PQ) - Pcalc(PQ);
    dQ = Qspec(PQ) - Qcalc(PQ);
    mis = [dP; dQ];

    if max(abs(mis)) < tol, break; end

    % ---------------- Jacobian blocks H, N, M, L ----------------
    H = zeros(npq); Nn = zeros(npq);       % Nn to avoid name clash
    M = zeros(npq); L = zeros(npq);

    for a = 1:npq
        i = PQ(a);
        % Diagonals
        % ∂P_i/∂θ_i = -Q_i - V_i^2*B_ii
        H(a,a) = -Qcalc(i);
        % ∂Q_i/∂θ_i =  P_i - V_i^2*G_ii
        M(a,a) =  Pcalc(i) - Vmag(i)^2 * G(i,i);
        % ∂P_i/∂|V_i| = 2*V_i*G_ii + Σ_{k≠i} V_k*(G_ik cosθik)
        sumN = 0; sumL = 0;
        for k = 1:N
            if k == i, continue; end
            dth = theta(i) - theta(k);
            sumN = sumN + Vmag(k)*G(i,k)*cos(dth);
            sumL = sumL + Vmag(k)*G(i,k)*sin(dth);
        end
        Nn(a,a) = 2*Vmag(i)*G(i,i) + sumN;
        % ∂Q_i/∂|V_i| = -2*V_i*B_ii + Σ_{k≠i} V_k*(G_ik sinθik)
        L(a,a)  =sumL;

        % Off-diagonals
        for b = 1:npq
            j = PQ(b); if j == i, continue; end
            dth = theta(i) - theta(j);
            % ∂P_i/∂θ_j
            H(a,b) =  Vmag(i)*Vmag(j)*G(i,j)*sin(dth);
            % ∂Q_i/∂θ_j
            M(a,b) = -Vmag(i)*Vmag(j)*G(i,j)*cos(dth);
            % ∂P_i/∂|V_j|
            Nn(a,b) =  Vmag(i)*G(i,j)*cos(dth);
            % ∂Q_i/∂|V_j|
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

% ---------------- Results ----------------
V = Vmag.* exp(1i*theta);
fprintf('Converged in %d iterations. Max mismatch = %.3e\n', it, max(abs(mis)));
for i = 1:N
    fprintf('Bus %d: |V| = %.6f pu, angle = %.6f deg\n', i, abs(V(i)), rad2deg(angle(V(i))));
end

% Branch currents
I12 = (V(1)-V(2))*y12;  I23 = (V(2)-V(3))*y23;  I14 = (V(1)-V(4))*y14;
fprintf('\nBranch currents (pu): I12=%.6f, I23=%.6f, I14=%.6f\n', abs(I12), abs(I23), abs(I14));

% Compliance checks
statRange = [0.96, 1.04]; Ith = 1.2;
within = (abs(V) >= statRange(1)) & (abs(V) <= statRange(2));
fprintf('\nVoltage compliance [%.2f, %.2f] pu: B1:%d B2:%d B3:%d B4:%d\n',...
        statRange(1), statRange(2), within(1), within(2), within(3), within(4));
fprintf('Currents <= %.2f pu: 1-2:%d  2-3:%d  1-4:%d\n', Ith, abs(I12)<=Ith, abs(I23)<=Ith, abs(I14)<=Ith);