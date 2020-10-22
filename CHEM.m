clc;
clear;
disp('Your Eq must be as forms : A+B--C+D etc');
in = input('Please input chemical Eq: \n','s');
Eq = in;
%Eq = 'S+O2--SO3';%2S+3O2==2SO3
%Eq = 'Na+H2O--NaOH+H2';
%Eq = 'C+O2--CO2';
%Eq = 'Fe+HCl--FeCl2+H2';
%Eq = 'H2+O2--H2O';
%Eq = 'MnO4+HCl--MnCl2+H2O+Cl2';%MnO4+8HCl==MnCl2+4H2O+3Cl2
EEq = '';
LsEq = '';
Elm = 0;
Mad = 1;
Elmarr=[];
Madarr=[];
Len = strlength(Eq);
for i = 1:Len
    Aski = 0;
    if Eq(i) == '+'
        Mad = Mad + 1;
    elseif Eq(i) == '-' && Eq(i+1) == '-'
        Mad = Mad + 1;
        i = i +1;
    end
    if Eq(i) >= 'A' && Eq(i) <= 'Z'
        for j = 1:strlength(EEq)
            if Eq(i) == EEq(j)
                Aski = 1;
                if (j+1) <= strlength(EEq)
                    if EEq(j+1) >= 'a' && EEq(j+1) <= 'z'
                        if Eq(i+1) ~= EEq(j+1)
                            Elm = Elm + 1;
                        end
                    end
                end
            end
        end
        if Aski == 0
            EEq = strcat(EEq,Eq(i));
            if Eq(i+1) >= 'a' && Eq(i+1) <= 'z'
                EEq = strcat(EEq,Eq(i+1));
                i = i +1;
            end
            Elm = Elm + 1;
        end
    end
end
Mtx = zeros(Elm,Mad);
Elm;%Element counter
Mad; %substance counter
Eq;%Cemical Equation
EEq;%Elemnts string
k = 1;
k_k = 0;
while k <= strlength(EEq)
    k_k = k_k + 1;
    Pre = EEq(k);
    if (k+1) <= strlength(EEq) && EEq(k+1) >= 'a'
        Pre = strcat(Pre,EEq(k+1));
        k = k+1;
    end
    if strlength(Pre) == 1
        l_l = 1;
        for i = 1:Len
            if Pre == Eq(i)
                if i+1 <= Len && Eq(i+1) <= '9' && Eq(i+1) >= '0' 
                    Mtx(k_k,l_l) = (Eq(i+1)-'0');
                else
                    Mtx(k_k,l_l) = 1;
                end
            end
            if Eq(i) == '+'
                l_l = l_l + 1;
            elseif (i+1) <= Len && Eq(i) == '-' && Eq(i+1) == '-'
                l_l = l_l + 1;
            end
        end
    else
        l_l = 1;
        for i = 1:Len
            if Pre(1) == Eq(i) && Pre(2) == Eq(i+1)
                if (i+2) <= Len && Eq(i+2) <= '9' && Eq(i+2) >= '0'
                    Mtx(k_k,l_l) = (Eq(i+2)-'0');
                else
                    Mtx(k_k,l_l) = 1;
                end
            end
            if Eq(i) == '+'
                l_l = l_l + 1;
            elseif (i+1) <= Len && Eq(i) == '-' && Eq(i+1) == '-'
                l_l = l_l + 1;
            end
        end
    end
    Pre = '';
    k = k + 1;
end
k = 1;
for i = 1:Len
    if Eq(i) == '-'
        break;
    end
    if Eq(i) == '+'
        k = k + 1;
    end
end
for i = 1:Elm
    for j = 1:Mad
        if j > k
            Mtx(i,j) = -1*Mtx(i,j);
        end
    end
end
Mtx;
% Substance matrix
b=zeros(Mad,1);
X = rref(Mtx,b);
Y=null(Mtx,'r');
%General Solution
t = size(Y);
t = t(1);
pm = 7200;
for i=2:t
    if pm > Y(i) && mod(Y(i)*1000,1000) ~= 0
        pm = Y(i);
    end
end
kop = 1;
for j = 1:7200
    if mod((pm*j)*1000,1000) == 0
       kop = j;
       break;
    end
end
if mod((pm*1000),1000) ~= 0
    for i=1:t
        Y(i) = Y(i)*kop;
    end
end
Y;
Cp = 1;
for i=1:Len
    Sff = Eq(i);
    if i == 1
        if Y(Cp) ~= 1
            LsEq = strcat(LsEq,(Y(Cp)+'0'));
        end
        Cp = Cp + 1;
    end
    if i > 1 && Eq(i-1) == '+'
        if Y(Cp) ~= 1
            LsEq = strcat(LsEq,(Y(Cp)+'0'));
        end
        Cp = Cp + 1;
    elseif i > 2 && Eq(i-1) == '-' && Eq(i-2) == '-'
        if Y(Cp) ~= 1
            LsEq = strcat(LsEq,(Y(Cp)+'0'));
        end
        Cp = Cp + 1;
    end
    if Sff == '-'
        Sff = '=';
    end
    LsEq = strcat(LsEq,Sff);
end
disp('Your input Chemical Equation is:');
Eq
disp('The trimed Chemical Equation is: ');
LsEq
