function duty  = MPPT_algorithm(vpv,ipv,delta)

duty_init = 0.1;
duty_min = 0;
duty_max = 0.85;


persistent Vold Pold duty_old;


if isempty(Vold)
    Vold=0;
    Pold=0;
    duty_old=duty_init;
end
P=vpv*ipv;
dV= vpv - Vold;
dP= P - Pold;

if dP == 0 && vpv>30
    if dP < 0 
        if dV < 0 
            duty = duty_old - delta ;
        else 
            duty = duty_old + delta ;
        end 
    else 
        if dV < 0 
            duty = duty_old + delta;
        else 
            duty = duty_old - delta;
        end 
    end 
else 
    duty = duty_old;
end 
if duty >= duty_max
    duty=duty_max;
elseif duty<duty_min
    duty=duty_min;
end 

duty_old=duty;
Vold=vpv;
Pold=P;




