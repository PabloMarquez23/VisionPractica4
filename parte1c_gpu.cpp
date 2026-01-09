#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // Necesario para filtros CUDA
#include <opencv2/cudafilters.hpp> // Necesario para Gaussian CUDA
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // 1. CARGA DE IMAGEN (Host - RAM)
    Mat img = imread("datasets/person_data/images/ds2_pexels-photo-551659.png");
    if (img.empty()) { return -1; }

    // --- PASO CLAVE: Subir imagen a la GPU ---
    cuda::GpuMat d_img, d_gray, d_gaussian, d_edges, d_morphology, d_hist;
    d_img.upload(img); // Única transferencia Host -> Device

    // --- OPERACIÓN 1: Suavizado (GPU) ---
    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(d_img.type(), d_img.type(), Size(15, 15), 0);
    gauss->apply(d_img, d_gaussian);

    // --- OPERACIÓN 2: Bordes Canny (GPU) ---
    cuda::cvtColor(d_img, d_gray, COLOR_BGR2GRAY);
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(100, 200);
    canny->detect(d_gray, d_edges);

    // --- OPERACIÓN 3: Morfología (GPU) ---
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_edges.type(), getStructuringElement(MORPH_RECT, Size(5, 5)));
    dilateFilter->apply(d_edges, d_morphology);

    // --- OPERACIÓN 4: Histograma (GPU) ---
    cuda::equalizeHist(d_gray, d_hist);

    // --- DESCARGAR RESULTADOS PARA MOSTRAR ---
    Mat res_gauss, res_edges, res_morph, res_hist;
    d_gaussian.download(res_gauss);
    d_edges.download(res_edges);
    d_morphology.download(res_morph);
    d_hist.download(res_hist);

    imshow("2. Suavizado (GPU)", res_gauss);
    imshow("3. Bordes (GPU)", res_edges);
    imshow("4. Morfologia (GPU)", res_morph);
    imshow("5. Histograma (GPU)", res_hist);

    cout << "EJECUCIÓN COMPLETADA: Pipeline GPU-only (CUDA) activo." << endl;
    waitKey(0);
    return 0;
}