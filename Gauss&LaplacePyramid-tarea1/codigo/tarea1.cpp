#define _DEBUG

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;

Mat convolution(Mat input, Mat mask)
{
	Mat output = input.clone();
	int anchoMask = mask.rows/2;//=2
	int altoMask = mask.cols/2;//=2
	Mat fondo = Mat::zeros(input.rows+2*anchoMask,input.cols+2*altoMask,CV_32FC1);//crear una matriz de ceros con bordes anchos
	//cout << input.rows << endl;
	for(int a=0; a<input.rows; a++)
	{ //este for es para rrellenar la matriz de ceros son la imagen
		for(int b=0; b<input.cols; b++)
		{
			fondo.at<float>(a+anchoMask,b+altoMask) = input.at<float>(a,b);
		}
	}
	
	for (int f=anchoMask; f<fondo.rows-anchoMask; f++)//este for es para aplicar la convolucion
	{
		for (int c=altoMask; c<fondo.cols-altoMask; c++)
		{
			float pixel=0.0;//estea variable es para el valor final del pixel
			/*cout << "pixel=" << pixel << endl;*/
			for(int mf=0; mf<mask.rows; mf++)
			{
				for (int mc=0; mc<mask.cols; mc++)	
				{
				/*cout << "f=" << f << ";  c=" << c <<";  mf=" << mf << ";  mc=" << mc << endl;*/
				pixel+=fondo.at<float>(f+mf-anchoMask,c+mc-altoMask)*mask.at<float>(mf,mc);
				//cout << "pixel=" << pixel << endl;
				}
			}
			output.at<float>(f-anchoMask,c-altoMask) = pixel;
		}
	}
    return output;
}

Mat compute_gauss_horiz(double sigma, int width)  // width debe ser impar
{
	//cout << "convolucion horizontal" << endl;
	Mat mask = Mat::zeros(1, width, CV_32FC1);
	// Por hacer: calcular mascara gaussiana horizontal pixel a pixel y normalizar la mascara resultante para que sume 1

	int anchoMask=mask.cols/2;
	float suma = 0.0;
	float mask_coef[width];

	for (int i = -anchoMask ; i<= anchoMask ; i++){
		//cout << "agarro primer for" << endl;
		//cout << "i=" << i << endl;
		mask_coef[i+anchoMask] = 1/exp((i*i)/(2*sigma*sigma));
		//cout << "mask_coef[i] = "<< mask_coef[i] << endl;
		suma = suma + exp(-(i*i)/(2*sigma*sigma));
	}
	//cout << "suma_horizontal=" << suma << endl;

	for (int c = 0 ; c < width ; c++){
		//cout << "agarro segundo for" << endl;
		//cout << "c=" << c << endl;
		mask.at<float>(0,c) = mask_coef[c]/suma;
		//cout << "mask.at<float>(0,c)" << mask.at<float>(0,c) <<endl;
	}
	return mask;
}

Mat compute_gauss_vert(double sigma, int height)
{
	//cout << "convolucion vertical" << endl;
	Mat mask = Mat::zeros(height, 1, CV_32FC1);
	// Por hacer: calcular mascara gaussiana vertical pixel a pixel y normalizar la mascara resultante para que sume 1	
	int altoMask=mask.rows/2;
	//cout << "altoMask=" << altoMask << endl;
	float suma = 0.0;
	float mask_coef[height];

	for (int i = -altoMask ; i<= altoMask ; i++){
		//cout << "agarro primer for" << endl;
		//cout << "i=" << i << endl;
		mask_coef[i+altoMask] = 1/exp((i*i)/(2*sigma*sigma));
		//cout << "mask_coef[i] = "<< mask_coef[i] << endl;
		suma = suma + exp(-(i*i)/(2*sigma*sigma));
	}
	//cout << "suma_vertical=" << suma << endl;

	for (int f = 0 ; f < height ; f++){
		//cout << "agarro segundo for" << endl;
		//cout << "f=" << f << endl;
		mask.at<float>(f,0) = mask_coef[f]/suma;
		//cout << "mask.at<float>(f,0)" << mask.at<float>(f,0) <<endl;
	}
	return mask;
}

Mat do_blur(Mat input, double sigma, int height)
{
	Mat result = input.clone(); // Esta linea debe ser eliminada/reemplazada
	// Por hacer:
	// (1) Calcular mascara horizontal con sigma 2.0 y ancho 7 usando funcion compute_gauss_horiz( )
	Mat mask_horiz;
	mask_horiz = compute_gauss_horiz(2,7);
	// (2) Calcular mascara vertical con sigma 2.0 y ancho 7 usando funcion compute_gauss_vert( )
	Mat mask_vert;
	mask_vert = compute_gauss_vert(2,7);
	// (3) Crear una nueva imagen haciendo convolucion con la mascara horizontal
	input = convolution(input , mask_horiz);
	// (4) Crear una nueva imagen haciendo convolucion con la mascara vertical
	result = convolution(input , mask_vert);
	// (5) devolver el resultado
	return result;
}

Mat subsample(Mat input)
{
	Mat result = Mat::zeros(input.rows/2,input.cols/2,CV_32FC1);	
	for (int f=0; f<result.rows; f++)
	{
		for (int c=0; c<result.cols; c++)
		{
			result.at<float>(f,c) = input.at<float>(2*f,2*c);
		}
	}
	return result;
}

vector<Mat> compute_gauss_pyramid(Mat input, int nlevels)
{
	vector<Mat> gausspyramid;//crea un arreglo dinamico de objetos Mat
	Mat current = input.clone();
	gausspyramid.push_back(current);//push_back debe ser un metodo de los arrglos llamado que mete en el arreglo un MAT
	//como al principio esta vacio es la primera que mete
	for (int i = 1; i < nlevels; i++)
	{
		// Por hacer:
		// (1) Aplicar do_blur( ) a la imagen gausspyramid[i-1], con sigma 2.0 y ancho 7
		// (2) Submuestrear la imagen resultante usando subsample( ), guardando el resultado en current
		Mat convolucionada, submuestreada;
		convolucionada = do_blur(gausspyramid[i-1],2.0,7);
		submuestreada = subsample(convolucionada);
		gausspyramid.push_back(submuestreada);
	}
	return gausspyramid;
}

void save_gauss_pyramid(vector<Mat> pyramid, const std::string& image_name)
{
	for (int i = 0; i < pyramid.size(); i++)
	{
		stringstream filename;
		filename << image_name << "_gauss_" << i << ".bmp";//para concatenar strings
		imwrite(filename.str(), pyramid[i]);
	}
}

/////

Mat subtract(Mat input1, Mat input2)
{
	// Esto es solo para verificar que las imagenes de entrada tengan el mismo tamano
	if (input1.cols != input2.cols || input1.rows != input2.rows)
	{
		cout << "subtract() called with different image sizes";
		return Mat();
	}
	// Por hacer: crear una nueva imagen que sea la resta entre input1 y input2
	Mat output = input1.clone();
	
	for(int f=0; f<input1.rows ; f++)
	{
		for(int c=0; c<input1.cols ; c++)
		{
		//cout<<"f="<<f<<endl;
		//cout<<"c="<<c<<endl;
		output.at<float>(f,c) = input1.at<float>(f,c) - input2.at<float>(f,c);
		//cout<<"aun no hay error"<<endl;	
		}
	}
	return output;
}

Mat add(Mat input1, Mat input2)
{
	// Esto es solo para verificar que las imagenes de entrada tengan el mismo tamano
	if (input1.cols != input2.cols || input1.rows != input2.rows)
	{
		cout << "add() called with different image sizes";
		return Mat();
	}
	// Por hacer: crear una nueva imagen que sea la suma entre input1 y input2
	Mat output = input1.clone();
	for(int f=0; f<input1.rows ; f++)
	{
		for(int c=0; c<input1.cols ; c++)
		{
		output.at<float>(f,c)=input1.at<float>(f,c) + input2.at<float>(f,c);
		}
	}
	return output;
	
	
}

Mat scale_abs(Mat input, float factor)//cambie double por float para que no choque la multiplicacion entre un pixel float y factor que antes era double 
{
	// Por hacer: aplicar valor absoluto a los pixeles de la imagen y luego escalar los pixeles usando el factor indicado
	Mat output = input.clone();

	for(int f=0 ; f<input.rows; f++)
	{
		for(int c=0; c<input.cols; c++)
		{
			output.at<float>(f,c) = abs(input.at<float>(f,c)) * factor;
		}
	}
	return output;
}

vector<Mat> compute_laplace_pyramid(Mat input, int nlevels)
{
	vector<Mat> gausspyramid;
	vector<Mat> laplacepyramid;
	Mat current = input.clone();
	Mat filtered;
	gausspyramid.push_back(current);
	for (int i = 1; i < nlevels; i++)
	{
		// Por hacer:
		// (1) Aplicar do_blur( ) a la imagen gausspyramid[i-1], con sigma 2.0 y ancho 7
		filtered = do_blur(gausspyramid[i-1],2.0,7);
		// (2) Guardar en laplacepiramid el resultado de restar gausspyramid[i - 1] y la imagen calculada en (1)
		Mat resultado_resta = subtract( gausspyramid[i - 1], filtered);
		laplacepyramid.push_back(resultado_resta); // Esta linea se debe reemplazar por lo indicado en (2)
		// (3) Submuestrear la imagen calculada en (1), guardar el resultado en current
		current = subsample(filtered);
		gausspyramid.push_back(current);
	}
	laplacepyramid.push_back(current);
	return laplacepyramid;
}

void save_laplace_pyramid(vector<Mat> pyramid, const std::string& image_name)
{
	// Este codigo debe basarse en el save_gauss_pyramid( )
	// Se debe escalar la intensidad de las imagenes antes de guardarlas usando: Mat scaled = scale_abs(pyramid[i], 5.0);
	// La ultima imagen de la piramide de Laplace no se debe escalar
	// El nombre de archivo debe comenzar con "laplace"
	for (int i = 0; i < pyramid.size(); i++)
	{
		Mat scaled = scale_abs(pyramid[i], 5.0);
		stringstream filename; //crea un objeto stringstream, flujo de strings, para concatenar strings 
		filename << image_name << "_laplace_" << i << ".bmp";// .bmp= bitmap, filename.str()=pasa el objeto stringstream a str
		imwrite(filename.str(), scaled);
	}
}

Mat upsample(Mat input)
{
	// Por hacer: implementar duplicacion del tama�o de imagen
	// Un pixel de la imagen de salida debe ser el promedio de los 4 pixeles mas cercanos de la imagen de entrada
	// Se debe tener cuidado de que los indices no salgan fuera del tamano de la imagen
	Mat output = Mat::zeros(input.rows*2,input.cols*2,CV_32FC1);
	for(int f=0 ; f<output.rows ; f++)
	{
		for(int c=0 ; c<output.cols ; c++)
		{
			/*
			cout<<"---------------------------------"<<endl;
			cout<<"c="<<c<<endl;
			cout<<"f="<<f<<endl;
			cout<<"input.rows="<<input.rows<<endl;
			cout<<"input.cols="<<input.cols<<endl;
			cout<<"output.rows="<<output.rows<<endl;
			cout<<"output.cols="<<output.cols<<endl;
			cout<<"entraron los for"<<endl;
			*/
			int col1, col2, row1, row2;
			float col1row1, col1row2, col2row1, col2row2;
			//cout<<"se crearon los int y floats"<<endl;
			col1 = int(c/2);
			if(c == output.cols-1){
				col2 = int (c/2);
			} else {
				col2 = int ((c+1)/2);
			}
			row1 = int (f/2);
			if(f == output.rows-1){
				row2 = int (f/2);	
			} else{
				row2 = int ((f+1)/2);
			}
			/*
			cout<<"col1="<<col1<<endl;
			cout<<"col2="<<col2<<endl;
			cout<<"row1="<<row1<<endl;
			cout<<"row2="<<row1<<endl;
			*/
			col1row1=input.at<float>(row1,col1);
			col1row2=input.at<float>(row2,col1);
			col2row1=input.at<float>(row1,col2);
			col2row2=input.at<float>(row2,col2);
			/*
			cout<<"col1row1="<<col1row1<<endl;
			cout<<"col1row2="<<col1row2<<endl;
			cout<<"col2row1="<<col2row1<<endl;
			cout<<"col2row2="<<col2row2<<endl;
			*/
			output.at<float>(f,c) = (col1row1 + col1row2 + col2row1 + col2row2)/ 4.0;
		}
	}
	return output;
}

Mat reconstruct(vector<Mat> laplacepyramid)
{
	Mat output = laplacepyramid[laplacepyramid.size()-1].clone();
	for (int i = 1; i < laplacepyramid.size(); i++)
	{
		int lev = int(laplacepyramid.size()) - i - 1;  // Nivel de la piramide de laplace, orden inverso
		// por hacer:
		// (1) Duplicar tamano output usando upsample( )
		Mat upsampled;
		upsampled = upsample(output); 
		// (2) Sumar resultado de (1) y laplacepyramid[lev] usando add( ), almacenar en output
		
		output = add(upsampled,laplacepyramid[lev]);
	}
	return output;
}

int main(void)
{
	Mat figurasllenasRGB = imread("figurasllenas.png");
	if (figurasllenasRGB.empty()) 
		{cout << "File not found" << endl;
		return 1;}
	Mat original_figurasllenas;
	cvtColor(figurasllenasRGB, original_figurasllenas, CV_BGR2GRAY);  
	Mat figurasLLenas;
	original_figurasllenas.convertTo(figurasLLenas, CV_32FC1); //nombre mat final: figurasLLenas
	
	Mat figurasbordeRGB = imread("figurasborde.png"); 
	if (figurasbordeRGB.empty()) 
		{cout << "File not found" << endl;
		return 1;}
	Mat original_figurasborde;
	cvtColor(figurasbordeRGB, original_figurasborde, CV_BGR2GRAY);  
	Mat figurasBorde;
	original_figurasborde.convertTo(figurasBorde, CV_32FC1);//nombre mat final: figurasBorde

	Mat monalisaRGB = imread("monalisa.png");
	if (monalisaRGB.empty()) 
		{cout << "File not found" << endl;
		return 1;}	
	Mat original_monalisa;
	cvtColor(monalisaRGB, original_monalisa, CV_BGR2GRAY);  
	Mat monalisa;
	original_monalisa.convertTo(monalisa, CV_32FC1); //nombre mat final: monalisa
	

	Mat cortezaRGB = imread("corteza.png");
	if (cortezaRGB.empty()) 
		{cout << "File not found" << endl;
		return 1;}
	Mat original_corteza;
	cvtColor(cortezaRGB, original_corteza, CV_BGR2GRAY);   
	Mat corteza;
	original_corteza.convertTo(corteza, CV_32FC1); //nombre mat final: corteza
	
	//test C.4
	Mat images_mat_names[4]={figurasLLenas, figurasBorde, monalisa, corteza};
	string images_str_names[4] = {"figurasllenas", "figurasborde" , "monalisa" , "corteza"}; 
	for (int i=0; i<4 ;i++)
	{
		vector<Mat> laplacepyramid = compute_laplace_pyramid(images_mat_names[i], 5);
		Mat reconstr = reconstruct(laplacepyramid);
		stringstream filename;
		filename << images_str_names[i] <<"_reconstr.bmp";//para concatenar strings
		imwrite(filename.str(), reconstr);
		
		Mat diferencia;
		diferencia = subtract(reconstr,images_mat_names[i]);
		stringstream filename_dif;
		filename_dif << images_str_names[i] <<"_diferencia.bmp";//para concatenar strings
		imwrite(filename_dif.str(),diferencia);
	}
	cout<<"-------------------------------------------------------------------"<<endl;
	cout<<"El codigo funciono y guardo todas las imagenes resultantes en la carpeta bluid."<<endl;	
	cout<<"-------------------------------------------------------------------"<<endl;
	/*
	//test C.3
	vector<Mat> laplacepyramid = compute_laplace_pyramid(monalisa, 5);
	Mat reconstr = reconstruct(laplacepyramid);
	
	Mat diferencia;
	diferencia = subtract(reconstr,monalisa);
	
	reconstr.convertTo(reconstr,CV_8UC1);
	imshow("reconstruida", reconstr);
	monalisa.convertTo(monalisa, CV_8UC1);
	imshow("original",monalisa);
	diferencia.convertTo(diferencia,CV_8UC1);
	imshow("diferencia",diferencia); 
	*/

	/*
	//test C.2
	Mat images_mat_names[4]={figurasLLenas, figurasBorde, monalisa, corteza};
	vector<Mat> laplacepyramid = compute_laplace_pyramid(monalisa, 5);
	Mat reconstruida;
	reconstruida = upsample(laplacepyramid[1]);
	reconstruida.convertTo(reconstruida, CV_8UC1);
	imshow("monalisa reconstruida",reconstruida);
	monalisa.convertTo(monalisa, CV_8UC1);
	imshow("monalisa original",monalisa);
    */

	/*
	//test B.4
	Mat images_mat_names[4]={figurasLLenas, figurasBorde, monalisa, corteza};
	string images_str_names[4] = {"figurasllenas", "figurasborde" , "monalisa" , "corteza"}; 
	for (int i=0; i<4 ;i++)
	{
		vector<Mat> laplacepyramid = compute_laplace_pyramid(images_mat_names[i], 5);
		save_laplace_pyramid(laplacepyramid,images_str_names[i]);
	}
	*/
		
	/*
	//test B.3
	//test abs
	Mat a= Mat::ones(500, 500, CV_8UC1)*255;
	imshow("esto deberia ser blanco",a);
	a.convertTo(a, CV_32FC1);
	a = a*(-1.0);
	Mat resultado_scale_abs = scale_abs(a,0.0);//esto es un test pasar una imagen de blanco a negro
	resultado_scale_abs.convertTo(resultado_scale_abs, CV_8UC1);
	imshow("la scale_abs esta bien si esto es negro",resultado_scale_abs);
	*/

	/*
	//test B.2
	vector<Mat> laplacepyramid = compute_laplace_pyramid(corteza, 5);
	Mat salida0, salida1, salida2, salida3, salida4;
	laplacepyramid[0].convertTo(salida0, CV_8UC1);
	imshow("laplacepyramind0",salida0);
	laplacepyramid[1].convertTo(salida1, CV_8UC1);
	imshow("laplacepyramind1",salida1);
	laplacepyramid[2].convertTo(salida2, CV_8UC1);
	imshow("laplacepyramind2",salida2);
	laplacepyramid[3].convertTo(salida3, CV_8UC1);
	imshow("laplacepyramind3",salida3);
	laplacepyramid[4].convertTo(salida4, CV_8UC1);
	imshow("laplacepyramind4",salida4);
	*/
	
	/*
	//test B.1
	Mat r1= Mat::ones(500, 500, CV_32FC1);
	Mat r2= Mat::ones(500, 500, CV_32FC1);
	Mat resultado_resta = subtract(r1,r2);
	resultado_resta.convertTo(resultado_resta, CV_8UC1);
	imshow("la resta esta bien si esto es negro",resultado_resta);
	*/

	//test A.6	
	/*
	Mat images_mat_names[4]={figurasLLenas, figurasBorde, monalisa, corteza};
	string images_str_names[4] = {"figurasllenas", "figurasborde" , "monalisa" , "corteza"}; 
	for (int i=0; i<4 ;i++)
	{
		vector<Mat> gausspyramid = compute_gauss_pyramid(images_mat_names[i], 5);
		save_gauss_pyramid(gausspyramid,images_str_names[i]);
	}
	*/
	
	/*
    // test A.5
	vector<Mat> gausspyramid = compute_gauss_pyramid(input, 5);
	Mat salida0, salida1, salida2, salida3, salida4;
	gausspyramid[0].convertTo(salida0, CV_8UC1);
	imshow("gausspyramind0",salida0);
	gausspyramid[1].convertTo(salida1, CV_8UC1);
	imshow("gausspyramind1",salida1);
	gausspyramid[2].convertTo(salida2, CV_8UC1);
	imshow("gausspyramind2",salida2);
	gausspyramid[3].convertTo(salida3, CV_8UC1);
	imshow("gausspyramind3",salida3);
	gausspyramid[4].convertTo(salida4, CV_8UC1);
	imshow("gausspyramind4",salida4);
	*/
	
	/*
	//test A.4
	Mat subsampled = subsample(output); 
	subsampled.convertTo(subsampled,CV_8UC1);
	imshow("subsample",subsampled);
	*/

	/* 
	//test A.3
	Mat output;
	output = do_blur(input,2,7);
	output.convertTo(output, CV_8UC1);
	imshow("filtrado",output);
	*/
	
	/*
	//Solucion problema de imshow una imagen de floats 32FC1
	double min,max;
	cv::minMaxLoc(output,&min,&max);
	output=output/max;
	imshow("filtrado",output);
	*/

	/*
	vector<Mat> gausspyramid = compute_gauss_pyramid(input, 5);
	save_gauss_pyramid(gausspyramid);

	vector<Mat> laplacepyramid = compute_laplace_pyramid(input, 5);
	save_laplace_pyramid(laplacepyramid);

	Mat reconstr = reconstruct(laplacepyramid);
	imwrite("reconstr.bmp", reconstr);	
	*/
	waitKey(0);
	return 0; // Sale del programa normalmente
}