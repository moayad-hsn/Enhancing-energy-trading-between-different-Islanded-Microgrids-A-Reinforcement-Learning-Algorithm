data=readtable("Day1.csv");

Tss=2.5e-6;

P=425000;
U=220;
f=50;
fsw=5e3;

Cfmax=(0.05*P)/(2*pi*f*U^2);
Increment_MPPT= 0.01;     % Increment value used to increase/decrease Vdc_ref
Limits_MPPT= [ 583 357 ];

timeStamp = data.TimeStamp;
irrad = data.SolarSensor_solar_irradiance_Avg;
temp = data.Hygro_Thermo_temperature_Avg;
windSpeed = data.TopAnemometer_wind_speed_Avg;
R=0.001;
L=27e-3;
C=13.5e-4;
Lf=(0.1*U^2)/(2*pi*f*P);
RLf=Lf*100;


