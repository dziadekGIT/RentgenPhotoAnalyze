#RentgenPhotoAnalyze - script for stacking X-ray photos of alloys.
Reading reflections from group of atoms from photo and fitting gaussian bell curve as FHMW values.

Picking point to analyze : 
<img width="637" alt="Zrzut ekranu 2024-04-26 o 12 34 29" src="https://github.com/dziadekGIT/RentgenPhotoAnalyze/assets/53622677/9f991cde-5376-47ed-a5d9-7547ee349dc1">

PyPlots line graph of number of photos (whitch are rotation angle of x-ray scan head) and brightess of pixels (which represents reflection of atom groups in particular rotation)
Next, script try to fit gaussian curve to pixel brightness and returns Full Hight Medium Width for location and plots it.
<img width="1501" alt="Zrzut ekranu 2024-04-26 o 12 35 47" src="https://github.com/dziadekGIT/RentgenPhotoAnalyze/assets/53622677/be3e19ca-96d1-47fa-89be-2de6fef085d8">

Scripts also can fit gaussian curve for all locations in picture size with process pool, and returns list of FWHM as json.
