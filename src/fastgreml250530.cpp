#if defined(__APPLE__) && defined(__aarch64__)
#  define EIGEN_USE_BLAS
#  define EIGEN_USE_LAPACKE
#else
#  define EIGEN_USE_MKL_ALL
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)    __builtin_expect(!!(x), 1)
#define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)    (x)
#define UNLIKELY(x)  (x)
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/resource.h>
#include <ctime>
#include <vector>

#include <omp.h>
#include <Eigen/Eigen>
#include <args.hxx>
#include <spdlog/spdlog.h>

#if defined(__linux__)
#include <fcntl.h>    // open
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close
#endif

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;
using Eigen::Vector2f;
//int n = 61923, m = 10000;
//int n = 11789, m = 17567;
//int n = 1967, m = 4929;
//int n = 98330, m = 10000;
//int n = 46727, m = 10000;
//int n = 170985, m = 10000;

int64_t ny, ngrm, n, r;
int C = 1;
int corenumber = 1;
int stepofconj = 25;
static default_random_engine e(time(0));
static normal_distribution<float> rnorm(0,1);
static std::uniform_real_distribution<float> runif(0, 1);
static std::uniform_real_distribution<float> runif2(-1, 1);
static std::random_device ran_device;
static std::mt19937 gen(ran_device());
static std::uniform_real_distribution<double> u_dis(0, 1);
//MatrixXf _X(n,m);
vector<float> yall, grmline;
vector<int> yid, grmid, yloc;
vector<bool> grmidwithy;
vector<int> nomissgrmid;
vector<int> grmloc;
bool _check = true;

struct Mphe {
    int i;
    std::string path;
};

std::vector<std::string> split_string(const std::string &str, char sep) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(sep);
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(sep, start);
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

bool compare_float(float a, float b, float epsilon = 1e-5f) {
  return std::fabs(a - b) < epsilon;
}

class PerfTimer {
public:
  explicit PerfTimer(const std::string &name = "")
      : name_(name), start_time_(std::chrono::high_resolution_clock::now()),
        elapsed_time_(start_time_), stopped_(false) {}

  ~PerfTimer() {
    if (!stopped_) {
      stop();
    }
    }

    void elapsed(const std::string& elapsed_name) {
        if (stopped_) return;  // Avoid double stop
        auto result = elapsed_to(elapsed_time_);
        double value = std::get<0>(result);
        std::string unit = std::get<1>(result);
        elapsed_time_ = std::get<2>(result);
        spdlog::info("[perf] ===> elapsed {}, cost {:.3f}{}", elapsed_name, value, unit);
    }

    void stop() {
        if (stopped_) return;  // Avoid double stop
        auto result = elapsed_to(start_time_);
        double value = std::get<0>(result);
        std::string unit = std::get<1>(result);
        spdlog::info("[perf] ===> {}, cost {:.3f}{}", name_, value, unit);
        stopped_ = true;
    }

private:
  std::tuple<double, std::string,
             std::chrono::high_resolution_clock::time_point>
  elapsed_to(const std::chrono::high_resolution_clock::time_point &time_point) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - time_point;
    std::chrono::duration<double, std::milli> milliseconds = duration;
    std::chrono::duration<double> seconds = duration;
    std::chrono::duration<double, std::ratio<60>> minutes = duration;
    std::chrono::duration<double, std::ratio<3600>> hours = duration;

    std::string unit = "ms";
    double value = milliseconds.count();

    if (milliseconds.count() < 1000.0) {
      value = milliseconds.count();
      unit = "ms";
    } else if (seconds.count() < 60.0) {
      value = seconds.count();
      unit = "s";
    } else if (minutes.count() < 60.0) {
      value = minutes.count();
      unit = "min";
    } else {
      value = hours.count();
      unit = "h";
    }

    return {value, unit, end_time};
  }

private:
    std::string name_;
    const std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point elapsed_time_;
    bool stopped_;
};


// Specialize formatter for Eigen matrices
namespace fmt {
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct formatter<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    // Parse format spec (we ignore it here)
    typename format_parse_context::iterator parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    // Format using operator<< to std::ostringstream
    template <typename FormatContext>
    typename FormatContext::iterator format(
        const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& mat,
        FormatContext& ctx) {

        std::ostringstream oss;
        oss << mat;
        return format_to(ctx.out(), "{}", oss.str());
    }
};

template <typename Derived>
struct formatter<Eigen::DenseBase<Derived>> {
    typename format_parse_context::iterator parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    typename FormatContext::iterator format(const Eigen::DenseBase<Derived>& expr,
                                            FormatContext& ctx) {
        std::ostringstream oss;
        oss << expr;
        return format_to(ctx.out(), "{}", oss.str());
    }
};

template <typename T>
struct formatter<T, char, typename std::enable_if<
    std::is_base_of<Eigen::EigenBase<T>, T>::value>::type>
{
    // No custom formatting, so ignore format spec
    typename format_parse_context::iterator parse(format_parse_context &ctx) {
      return ctx.begin();
    }

    // Format Eigen object by streaming to std::ostringstream
    template <typename FormatContext>
    typename FormatContext::iterator format(const T& mat, FormatContext& ctx) {
        std::ostringstream oss;
        oss << mat; // Uses Eigen's operator<<
        return format_to(ctx.out(), "{}", oss.str());
    }
};
} // namespace fmt

struct Grmbin{  //230512
    MatrixXf A, B, CmatA;
    VectorXf diag, remle, remlreg;
    float yAy;
};

vector<Grmbin> _singlebin;

struct Lseed{
    int j;
    MatrixXf U;
    VectorXf beta, delta, rho;
};

MatrixXf _grm(n,n);
Eigen::SparseMatrix<float, Eigen::RowMajor> _grmsp(n,n);
VectorXf _grmline(n);
MatrixXf _A(1,1);
MatrixXf _B(1,1);
VectorXf _diag(n);
VectorXf _y(n);
vector<int> _J;
MatrixXf _Cmat(n,C);
MatrixXf _CmatA(n,C);
MatrixXf _XXCmat(C,n);
MatrixXf _C(1,1);
MatrixXd _XXdinv(C,C);
//VectorXi _landmark(n);

//230512
MatrixXf _StoC(1,1);
VectorXf mhe(1);
VectorXf remlestc(1);
VectorXf remlstc(1);
MatrixXf mhes(1,1);

vector<long long> _landmark;
//MatrixXf _grmc(n,n);
VectorXf _yc(n);
//VectorXf _grmline(n*(n+1)/2);
//vector<MatrixXf> _A(2);
Eigen::Vector2f _u(0.5,0.5);

//MatrixXf I = MatrixXf::Identity(sn,sn);

int rand_seed()
{
    stringstream str_strm;
    str_strm<<time(NULL);
    string seed_str=str_strm.str();
    reverse(seed_str.begin(), seed_str.end());
    seed_str.erase(seed_str.begin()+7, seed_str.end());
    return(abs(atoi(seed_str.c_str())));
}


double ran1(int &idum) {
    return u_dis(gen);
}

double gasdev(int &idum) {
    static int iset = 0;
    static double gset;
    double fac, rsq, v1, v2;

    if (idum < 0) iset = 0;
    if (iset == 0) {
        do {
            v1 = 2.0 * ran1(idum) - 1.0;
            v2 = 2.0 * ran1(idum) - 1.0;
            rsq = v1 * v1 + v2*v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1*fac;
        iset = 1;
        return v2*fac;
    } else {
        iset = 0;
        return gset;
    }
}



//VectorXf conjugate(MatrixXf V, VectorXf y, int step){
//    VectorXf b = y;
//    VectorXf xk = b;
//    VectorXf rk = b - V * xk; //initial guess of Viy is y, actually can set 0
//    VectorXf pk = rk;
//    float rkrk, ak, bk = 0.0;
//    VectorXf Apk(n);
//    //VectorXf res(step);
//    for(int i = 0; i < step; i++){
//        rkrk = rk.squaredNorm();
////        if(rkrk == 0.0){
////            return xk;
////        }
//        Apk = V * pk;
//        ak = rkrk / Apk.dot(pk); //step size
//        xk += ak * pk;  //solution
//        rk -= ak * Apk; //residual
//        bk = rk.squaredNorm() / rkrk; //determines the contribution of the previous search direction
//        pk = rk + bk * pk; //calculate conjugate direction
//        //res(i) = pk.array().abs().sum();
//    }
//    return xk;
//}
bool isBlankLine(const std::string& line) {
    for (char c : line) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

int countValidLines(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return 0; // fail to open
    }

    int validCount = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (!isBlankLine(line)) {
            ++validCount;
        }
    }

    return validCount;
}


int countItemnumber(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return 0; // fail to open
    }
    std::string firstLine;
    if (!std::getline(file, firstLine)) {
        std::cerr << "Error: File is empty\n";
        return 0;
    }
    std::istringstream iss(firstLine);
    std::string token;
    int count = 0;
    while (iss >> token) {
        count++;
    }
    return count;
}

string getOrdinal(int num) {
    int lastTwo = num % 100;
    if (lastTwo >= 11 && lastTwo <= 13) {
        return to_string(num) + "th";
    }
    
    int lastDigit = num % 10;
    switch (lastDigit) {
        case 1: return to_string(num) + "st";
        case 2: return to_string(num) + "nd";
        case 3: return to_string(num) + "rd";
        default: return to_string(num) + "th";
    }
}

float Hecal(MatrixXf gtemp, VectorXf ytemp){
    VectorXf _Ay = gtemp * ytemp;
    float _yAy = _Ay.dot(ytemp);
    return (_yAy - n)/(gtemp.squaredNorm() - n);
}

float HEofGCTA(MatrixXf gtemp, VectorXf ytemp){
    VectorXf _Ay = gtemp * ytemp;
    float _yAy = _Ay.dot(ytemp);
    cout << _yAy <<endl;
    VectorXf Ayii = (ytemp.array() * ytemp.array()).matrix();
    return (_yAy - Ayii.dot(gtemp.diagonal()) - 1)/(gtemp.squaredNorm() - gtemp.diagonal().squaredNorm() - 1);
}

float Hed(MatrixXf &gtemp, VectorXf ytemp){
    VectorXf grmd = gtemp.diagonal();
    MatrixXf grmhe = grmd.asDiagonal();
    MatrixXd grmhed = (gtemp - grmhe).cast<double>();
    VectorXd Ay = grmhed * ytemp.cast<double>();
    double yAy = Ay.dot(ytemp.cast<double>());
    double he = (yAy - 1.0)/(grmhed.squaredNorm() - 1.0);
    return (float)(he);
}

float Remlone(MatrixXf gtemp, VectorXf ytemp){
    VectorXf Ay = gtemp * ytemp;
    float yAy = Ay.dot(ytemp);
    float yAAy = Ay.squaredNorm();
    Eigen::Matrix2f one;
    one << yAAy, yAy, yAy, n;
    return - one.inverse().col(0).sum() * n;
}




void read_realphe_all(string phefile){
    yall.clear(); yid.clear();
    string pheitem,s;
    ifstream phe(phefile, ios::in);
    if (!phe.is_open()) cout << "can not open the file phe\n";
    while (!phe.eof()){
        getline(phe,pheitem);
        istringstream is(pheitem);
        is >> s; is >> s; yid.push_back(stoi(s));
        is >> s; yall.push_back(stof(s));
    }
    phe.close();
    ny = yid.size();
}

void read_grmid(string grmidfile) {
    grmid.clear();
    string grm_id_file_path = grmidfile + ".grm.id";
    spdlog::info("Reading grm.id file from [{}]", grm_id_file_path);
    string line;
    ifstream fin(grm_id_file_path, ios::in);
    if (!fin.is_open()) spdlog::error("can not open the grmid file");
    getline(fin, line);
    while (!fin.eof()){
        grmid.push_back(stoi(line));
        getline(fin,line);
    }
    fin.close();
    ngrm = grmid.size();
    grmidwithy.resize(ngrm, false);
    spdlog::info("The number of individuals included in GRM is: {}", ngrm);
}

void read_grmline(string grmfile){
    clock_t start, end;
    double totaltime, cumutime;
    long long grmsize = (long long)ngrm * ((long long)ngrm + 1)/2;
    grmline.clear();
    grmline.reserve(grmsize);
    //cout << grmline.size() <<endl;
    //cout << grmline.capacity() <<endl;
    long long count = 1;
    ifstream fin( grmfile, ios::in | ios::binary);
    if (!fin.is_open()) spdlog::error("can not open the file");
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.read((char*) &f_buf, size);
    //end = clock();
    //cout <<  (double)(end - start) / CLOCKS_PER_SEC << endl;
    start = clock();
    while (!fin.eof()){
         count ++;
        if(count % 1000000000 == 0) {
            end = clock();
            if(count == 1000000000){
                totaltime  = (double)(grmsize / 1000000000) * (double)(end - start) / CLOCKS_PER_SEC;
            }
            cumutime = (double)(end - start) / CLOCKS_PER_SEC;
            spdlog::info("grm data reading process: {}s/{}s", ceil(cumutime), ceil(totaltime));
        }
        grmline.push_back(f_buf);
        fin.read((char*) &f_buf, size);
    }
    fin.close();
    //cout << grmline.size() << endl;
    if (grmline.size() != (long long)ngrm * ((long long)ngrm + 1)/2)
        spdlog::info("grmfile and grmid don't match");
}


VectorXf merge_pheid_grmid(){
    int i = 0, j = 0;
    yloc.clear(); grmloc.clear();
    while(i < ny && j < ngrm){
        if(yid[i] < grmid[j]) i++;
        else if (yid[i] > grmid[j]) j++;
        else {
            yloc.push_back(i); grmloc.push_back(j);
            i++; j++;
            //cout << grmid[grmloc[j]] << endl;
        }
    }
    n = yloc.size();
    VectorXf y = VectorXf::Zero(n);
    for(i = 0; i < n; i++){
        y(i) = yall[yloc[i]];
    }
    yloc.clear();
    y -= VectorXf::Constant(n, y.mean());
    y /= (y.norm() / sqrt(n - 1));
    
    return y;
}

void read_realcov_ofcommonid(string covfile){
    _Cmat.resize(n, C);
    //MatrixXf cov = MatrixXf::Zero(n, C);
    string index,covitem,s;
    int i,j,temp,indextemp;
    ifstream covin(covfile, ios::in );
    if (!covin.is_open()) spdlog::info("can not open the covfile");
    for (i = 0; i < n; i++) {
        indextemp = grmid[grmloc[i]]; //the first number in each line of grmfile: id
        do {
            getline(covin, covitem);
            temp = stoi(covitem);
        } while (indextemp != temp);
        istringstream is(covitem);
        is >> s; is >> s;
        for(j = 0; j < C; j++){
            is >> s;
            _Cmat(i,j) = stof(s);
        }
        _Cmat(i,C - 1) = 1.0;
    }
    covin.close();
    //MatrixXf XX = _Cmat.transpose() * _Cmat;
    MatrixXd XXd = _Cmat.cast<double>().transpose() * _Cmat.cast<double>();
    MatrixXd Imatd = MatrixXd::Identity(C,C);
    //MatrixXd XXd = XX.cast<double>();
    _XXdinv = XXd.ldlt().solve(Imatd);
    //_XXCmat = _XXdinv.cast<float>() * _Cmat.transpose();
    _XXCmat = (_XXdinv * _Cmat.transpose().cast<double>()).cast<float>();
}



void read_grm(string file){
    _grm = MatrixXf::Zero(n, n);
    int i = 0, j = 0;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) spdlog::info("can not open the file");
    int size = sizeof (float);
    float f_buf = 0.0;
    for (i = 0; i < n; i++) {
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        for (j = 0; j <= i; j++) {
          fin.read((char *)&f_buf, size);
          _grm(j, i) = _grm(i, j) = f_buf;
          // _grm(i, j) = f_buf;
        }
    }
    fin.close();
}



void read_grmAB(string file){
    int halfn = (n + 1)/2;
    _A.setZero(halfn, halfn);
    _B.setZero(halfn, halfn);
    _diag.setZero(n);
    int i = 0, j = 0;
    long long rowloc = 0, loc = 0;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    for(i = 0; i< halfn; i++){
        //fin.seekg((_landmark[i] + (long long)grmloc[j]) * size,ios::beg);
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else _A(i,j) = f_buf;
        }
    }
    for(i = halfn; i< n; i++){
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(j = 0; j<= i; j++){
           // fin.seekg((_landmark[i] + (long long)grmloc[j]) * size,ios::beg);
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else if(j < halfn) _B(i - halfn,j) = f_buf;
            else _A(j - halfn, i - halfn) = f_buf;
        }
    }
    fin.close();
}

void read_grm_Ai(int i, string file){
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float), loc;
    float f_buf = 0.0;
    fin.seekg((_landmark[i] + (long long)grmloc[0]) * size,ios::beg);
    if(i % 10000 == 0) cout << "Reading id:" << i << endl;
    for(int j = 0; j<= i; j++){
        fin.read((char*) &f_buf, size);
        if(i == j) _diag(i) = f_buf;
        else _A(i,j) = f_buf;
        loc = grmloc[j + 1] - grmloc[j] - 1;
        if(loc != 0)
        fin.seekg((loc) * size,ios::cur);
    }
    fin.close();
}

void read_grm_Bi(int i, string file){
    int halfn = (n + 1)/2;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float), loc;
    float f_buf = 0.0;
    fin.seekg((_landmark[i] + (long long)grmloc[0]) * size,ios::beg);
    if(i % 10000 == 0) cout << "Reading id:" << i << endl;
    for(int j = 0; j<= i; j++){
      fin.read((char*) &f_buf, size);
      if(i == j) _diag(i) = f_buf;
      else if(j < halfn) _B(i - halfn,j) = f_buf;
      else _A(j - halfn, i - halfn) = f_buf;
      loc = grmloc[j + 1] - grmloc[j] - 1;
      if(loc != 0)
      fin.seekg((loc) * size,ios::cur);
    }
    fin.close();
}

//void read_grmABmiss(string file){
//        int halfn = (n + 1)/2;
//        _A.setZero(halfn, halfn);
//        _B.setZero(halfn, halfn);
//        _diag.setZero(n);
//    int i;
//#pragma omp parallel for schedule(dynamic, 1)
//    for(i = 0; i< halfn; i++){
//        read_grm_Ai(i, file);
//    }
//#pragma omp parallel for schedule(dynamic, 1)
//    for(i = halfn; i< n; i++){
//        read_grm_Bi(i, file);
//    }
//}


void read_grmABmiss(string file){
    int halfn = (n + 1)/2;
    _A.setZero(halfn, halfn);
    _B.setZero(halfn, halfn);
    _diag.setZero(n);
    int i = 0, j = 0, colloc = grmloc[0], loc;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
#pragma omp parallel for schedule(dynamic, 1)
    for(i = 0; i< halfn; i++){
        fin.seekg((_landmark[i] + (long long)grmloc[0]) * size,ios::beg);
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else _A(i,j) = f_buf;
            loc = grmloc[j + 1] - grmloc[j] - 1;
            if(loc != 0)
            fin.seekg((loc) * size,ios::cur);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for(i = halfn; i< n; i++){
        fin.seekg((_landmark[i] + (long long)grmloc[0]) * size,ios::beg);
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else if(j < halfn) _B(i - halfn,j) = f_buf;
            else _A(j - halfn, i - halfn) = f_buf;
            loc = grmloc[j + 1] - grmloc[j] - 1;
            if(loc != 0)
            fin.seekg((loc) * size,ios::cur);
        }
    }
    fin.close();
}


void readgrm(int timemethod, string grmfile){
    _landmark.clear();
    _landmark.reserve(n);
    //_J.clear();
     for (int i = 0; i< n; i++){
        //_J.push_back(i);
        _landmark.push_back((long long)(grmloc[i]) * ( (long long)(grmloc[i]) + 1) / 2);
     }
    
    string grmfilefull = grmfile + ".grm.bin";
    if (timemethod == 0)
        read_grm(grmfilefull);
    if (timemethod == 1)
        read_grmABmiss(grmfilefull);
    //if (timemethod == 2)
      //  read_grm_lower(grmfilefull);
    else if (timemethod == 3){
        read_grmline(grmfilefull);
           
    }

}


VectorXf read_realphe(string grmid, string phefile){
    VectorXf y = VectorXf::Zero(n);
    string index,pheitem,s;
    int i,temp,indextemp;
    ifstream fin(grmid, ios::in );
    ifstream phe(phefile, ios::in );
    if (!fin.is_open()) cout << "can not open the id list\n";
    if (!phe.is_open()) cout << "can not open the file phe\n";
    for (i = 0; i < n; i++) {
        getline(fin,index);
        indextemp = stoi(index); //the first number in each line of grmfile: id
        do {
            getline(phe,pheitem);
            temp = stoi(pheitem);
        } while (indextemp != temp);
        istringstream is(pheitem);
        is >> s; is >> s; is >> s;
        y(i) = stof(s);
    }
    fin.close();
    phe.close();
    
    y -= VectorXf::Constant(n, y.mean());
    y /= (y.norm() / sqrt(n - 1));
    //cout <<  y.head(5) <<endl;
    return y;
}

MatrixXf read_realcov(string grmid, string covfile){
    MatrixXf cov = MatrixXf::Zero(n, C);
    string index,covitem,s;
    int i,j,temp,indextemp;
    ifstream fin(grmid, ios::in );
    ifstream covin(covfile, ios::in );
    if (!fin.is_open()) cout << "can not open the id list\n";
    if (!covin.is_open()) cout << "can not open the file phe\n";
    for (i = 0; i < n; i++) {
        getline(fin,index);
        indextemp = stoi(index); //the first number in each line of grmfile: id
        do {
            getline(covin, covitem);
            temp = stoi(covitem);
        } while (indextemp != temp);
        istringstream is(covitem);
        is >> s; is >> s;
        for(j = 0; j < C; j++){
            is >> s;
            cov(i,j) = stof(s);
        }
        cov(i,C - 1) = 1.0;
    }
    fin.close();
    covin.close();
//    for(j = 0; j < C; j++){  //standardised Cmat
//        cov.col(j) -= VectorXf::Constant(n, cov.col(j).mean());
//        cov.col(j) /= (cov.col(j).norm() / sqrt(n - 1));
//    }
    return cov;
}



//VectorXf proj_x(VectorXf x){
//    VectorXf XXCmatx = _XXCmat * x;
//    x -= _Cmat * XXCmatx;
//    return x;
//}

VectorXf proj_x(const VectorXf& x) {
    
    //VectorXd XXCmatx = _XXdinv * _Cmat.cast<double>().transpose() * x.cast<double>();
    //VectorXd XXCmatx = _XXCmat.cast<double>() * x.cast<double>();
    //return (x.cast<double>() - _Cmat.cast<double>() * XXCmatx).cast<float>();
    //VectorXf XXCmatx = _XXCmat * x;
    //return x - _Cmat * XXCmatx;
    return x - _Cmat * (_XXCmat * x);
}


float grmlinetimesvectorparallel(int i, const VectorXf& x){
    float element = 0.0;
    int j = 0;
        for (j = 0; j< i; j++){
           // pos = grmloc[i] * ( grmloc[i] + 1) / 2 + grmloc[j];
            //pos = _landmark[i] + grmloc[j];
           // element += grmline[pos] * x(j);
            element += grmline[_landmark[i] + (long long)grmloc[j]] * x(j);
           // element += grmline[ grmloc[i] * ( grmloc[i] + 1) / 2 + grmloc[j]] * x(j);
            //if(i == 0 && j < 5) cout << grmline[_landmark[j] + grmloc[i]] << endl;
        }
        for (j = i; j< n; j++){
            //pos = grmloc[j] * ( grmloc[j] + 1) / 2 + grmloc[i];
            //pos = _landmark[j] + grmloc[i];
            //element += grmline[pos] * x(j);
            element += grmline[_landmark[j] + (long long)grmloc[i]] * x(j);
          //  if(i == 0 && j < 5) cout << grmline[_landmark[j] + grmloc[i]] << endl;
        }
    return element;
}

VectorXf grmlinetimesvector(const VectorXf& x){
    VectorXf Ax(n);
    int i = 0;
    // omp_set_num_threads(phenumber);
     #pragma omp parallel for
    for (i = 0; i< n; i++){
        Ax(i) = grmlinetimesvectorparallel(i, x);
    }
    return Ax;
}

float grmlinetimesvectorparallel2(int i, const VectorXf& x){
    VectorXf temp(n);
    int j = 0;
        for (j = 0; j< i; j++){
           // pos = grmloc[i] * ( grmloc[i] + 1) / 2 + grmloc[j];
            //pos = _landmark[i] + grmloc[j];
           // element += grmline[pos] * x(j);
            temp(j) = grmline[_landmark[i] + (long long)grmloc[j]];
           // element += grmline[ grmloc[i] * ( grmloc[i] + 1) / 2 + grmloc[j]] * x(j);
            //if(i == 0 && j < 5) cout << grmline[_landmark[j] + grmloc[i]] << endl;
        }
        for (j = i; j< n; j++){
            //pos = grmloc[j] * ( grmloc[j] + 1) / 2 + grmloc[i];
            //pos = _landmark[j] + grmloc[i];
            //element += grmline[pos] * x(j);
            temp(j) = grmline[_landmark[j] + (long long)grmloc[i]];
          //  if(i == 0 && j < 5) cout << grmline[_landmark[j] + grmloc[i]] << endl;
        }
    return temp.dot(x);
}

VectorXf grmlinetimesvector2(const VectorXf& x) {
    VectorXf Ax(n);
    int i = 0;
   // omp_set_num_threads(phenumber);
    #pragma omp parallel for
    for (i = 0; i< n; i++){
        Ax(i) = grmlinetimesvectorparallel2(i, x);
    }
    return Ax;
}


VectorXf grmlinetimesvectorgpt(VectorXf x){
    VectorXf Ax(n);
    int i = 0, j = 0;
    float element = 0.0;
    for (i = 0; i< n; i++){
        element = std::accumulate(_J.begin(), _J.end(), 0.0,
            [&](float sum, int j){
                return sum + (j < i ? grmline[_landmark[i] + grmloc[j]] : grmline[_landmark[j] + grmloc[i]]) * x(j);
            });
        Ax(i) = element;
    }
    return Ax;
}


VectorXf grmABtimesvector(const VectorXf& x){
    int halfn;
    if(n % 2 == 0){
        halfn = n/2;
        const auto& y1 = x.head(halfn);
        const auto& y2 = x.tail(halfn);
        VectorXf Vy(n);
        Vy.head(halfn).noalias() = _A.selfadjointView<Eigen::Lower>() * y1 + _B.transpose() * y2;
        Vy.tail(halfn).noalias() = _A.selfadjointView<Eigen::Upper>() * y2 + _B * y1;
        return Vy.array() + _diag.array() * x.array();
    }
    else{
        halfn = (n + 1)/2;
        VectorXf xpan(n + 1);
        xpan << x, 0;
        const auto& y1 = xpan.head(halfn);
        const auto& y2 = xpan.tail(halfn);
        VectorXf Vy(n + 1);
        Vy.head(halfn).noalias() = _A.selfadjointView<Eigen::Lower>() * y1 + _B.transpose() * y2;
        Vy.tail(halfn).noalias() = _A.selfadjointView<Eigen::Upper>() * y2 + _B * y1;
        return Vy.head(n).array() + _diag.array() * x.array();
    }

}


VectorXd grmABtimesvectorddd(VectorXd x){
    int halfn;
    if(n % 2 == 0){
        halfn = n/2;
        VectorXd y1 = x.head(halfn);
        VectorXd y2 = x.tail(halfn);
        VectorXd Vy1 = _A.cast<double>().selfadjointView<Eigen::Lower>() * y1 + _B.cast<double>().transpose() * y2;
        VectorXd Vy2 = _A.cast<double>().selfadjointView<Eigen::Upper>() * y2 + _B.cast<double>() * y1;
        VectorXd Vy(n);
        Vy << Vy1, Vy2;
        return Vy + _diag.cast<double>().cwiseProduct(x);
    }
    else{
        halfn = (n + 1)/2;
        VectorXd xpan(n + 1);
        xpan << x, 0;
        VectorXd y1 = xpan.head(halfn);
        VectorXd y2 = xpan.tail(halfn);
        VectorXd Vy1 = _A.cast<double>().selfadjointView<Eigen::Lower>() * y1 + _B.cast<double>().transpose() * y2;
        VectorXd Vy2 = _A.cast<double>().selfadjointView<Eigen::Upper>() * y2 + _B.cast<double>() * y1;
        VectorXd Vy(n + 1);
        Vy << Vy1, Vy2;
        return Vy.head(n) + _diag.cast<double>().cwiseProduct(x);
    }

}


VectorXf Actimesx(const VectorXf& x, int cho){
    if (cho  == 0)
    return proj_x(_grm * proj_x(x));
    else if (cho == 1)
        return proj_x(grmABtimesvector(proj_x(x)));
        //return proj_x(grmABtimesvectorddd(proj_x(x).cast<double>()).cast<float>());
    //else if (cho == 2)
       // return proj_x(_grmsp.selfadjointView<Eigen::Lower>() * proj_x(x));
    else if (cho == 3)
        return proj_x(grmlinetimesvector(proj_x(x)));
    else
        return proj_x(grmlinetimesvector2(proj_x(x)));
    //return proj_x(grmlinetimesvectorgpt(proj_x(x)));
}


VectorXf trgrmc(){
    _CmatA.resize(C,n);  // x'A
    float xxxxA, xxxxAA, term1, term4;
    for (int i = 0; i < C; i++){
        _CmatA.row(i) = grmABtimesvector(_Cmat.col(i));
    }
    term1 = (float)((_A.cast<double>().squaredNorm()+_B.cast<double>().squaredNorm()) * 2 + _diag.cast<double>().squaredNorm());
    xxxxA = (_CmatA.cwiseProduct(_XXCmat)).sum();
    xxxxAA = (_CmatA * _CmatA.transpose()).cwiseProduct(_XXdinv.cast<float>()).sum();
    MatrixXf xxxAx = _CmatA * _XXCmat.transpose();
    term4 = (xxxAx * xxxAx).diagonal().sum();
    VectorXf tr(2);
    tr << _diag.sum() - xxxxA, term1 - xxxxAA * 2 + term4;
    return tr;
}


VectorXf Hec(VectorXf y, int numc){
  //  VectorXf Ay = proj_x(grmABtimesvector(y));
    VectorXf Ay = proj_x(grmABtimesvector(y));
    float yAy = Ay.dot(y);
    VectorXf trs = trgrmc();
    float trAcAc = trs(1), trAc = trs(0);
    MatrixXf Hi(2,2);
    Hi << n - numc + 0.0, -trAc, -trAc, trAcAc;
    Hi /= (trAcAc * (n - numc) - trAc * trAc);
    VectorXf R(2), he;
    R << yAy, n - 1.0;
    he = Hi * R;
    VectorXf he_trAc(2);
    he_trAc << he(0) / he.sum(), trAc;
    return he_trAc;
}

VectorXf conjugate(float varcmp, float ve, const VectorXf& x, int timemethod){
    VectorXf Apk(n);
    const auto& bb = x;
    VectorXf xk = bb;
    VectorXf rk = bb - (varcmp * Actimesx(xk,timemethod) + ve * xk) ;
    VectorXf pk = rk;
    float rkrk, ak, yy, bk = 0.0;
    yy = bb.dot(bb);
    for(int i = 0; i < stepofconj; i++){
        rkrk = rk.squaredNorm();
        if((rkrk / yy) < 1e-12) break;
        Apk.noalias() = (varcmp * Actimesx(pk,timemethod) + ve * pk);
        ak = rkrk / Apk.dot(pk);
        xk += ak * pk;
        // if(_check){
        //     cout << "   step" <<  i << ": " << pk.array().abs().sum()*ak <<endl;
        // }
        rk -= ak * Apk;
        bk = rk.squaredNorm() / rkrk;
        pk = rk + bk * pk;
        //res(i) = pk.array().abs().sum();
    }
    return xk;
}

VectorXf Vtimesx(float varcmp,float ve, VectorXf x){
    //20230505
    return (varcmp * Actimesx(x,1) + ve * x);
}


Lseed lanczos_seed(float varcmp, VectorXf x){
    float tol = 2.0, ve = 1.0 - varcmp, res_norm = 1.0; //its just enough for 50k
    int j = 0, maxit = 50;
    bool cnvg = false;
    MatrixXf U(n,maxit);
    VectorXf R(n),Vv(n), beta(maxit), rho(maxit), omega(maxit), delta(maxit), gamma(maxit);
    U.setZero(); Vv.setZero();R.setZero(); beta.setZero(); rho.setZero(); omega.setZero(); delta.setZero(); gamma.setOnes();

    // Initial values
    beta(0) = rho(0) = x.norm();
    U.col(0) = x / beta(0); //v1 = arbitrary norm1 vector

   // while(j < maxit){
    while(j < maxit - 1){   //20240114
    // Lanczos iteration
        Vv = Vtimesx(varcmp, ve, U.col(j));
        delta(j) = U.col(j).dot(Vv);
        if(j == 0) Vv = Vv - delta(j) * U.col(j);
        else Vv = Vv - delta(j) * U.col(j) - beta(j) * U.col(j-1);
        beta(j+1) = Vv.norm();
        U.col(j+1) = Vv / beta(j+1);

        // CG coefficients update
        if(j == 0) gamma(j) = 1.0 / delta(j);
        else gamma(j) = 1.0 / (delta(j) - omega(j-1) / gamma(j-1));
        omega(j) = beta(j+1) * gamma(j) * beta(j+1) * gamma(j);
        rho(j+1) = -beta(j+1) * gamma(j) * rho(j);

        // CG vectors update
        R = rho(j+1) * U.col(j+1);
        j++;
        res_norm = R.norm();
        //cout << "Error at step " << j << " is " << res_norm << endl;
        if(res_norm <= tol){
          cnvg = true;
          break;
        }
  }
   // if(cnvg) cout << "Converged after " << j << " iterations." << endl;
   // else cout << "Failed to converge after " << j << " iterations." << endl;

    Vv = Vtimesx(varcmp, ve, U.col(j));
    delta(j) = U.col(j).dot(Vv);
    
    Lseed seed;
    seed.j = j; seed.U = U.leftCols(j); seed.beta = beta.head(j); seed.delta = delta.head(j); seed.rho = rho.head(j);
    return seed;
}

VectorXf lanczos_solve(Lseed seed, VectorXf x, float sigma){
    float tol = 5e-1,res_norm = 1.0; //its just enough for 50k
   // MatrixXf U = seed.U; VectorXf beta = seed.beta, delta = seed.delta, rho = seed.rho;
    MatrixXf U = seed.U; VectorXf beta = seed.beta, delta = seed.delta, rho = delta; rho(0) = x.norm();
    delta.array() += sigma;
    int maxj = seed.j;
    VectorXf R(n), X(n), P(n), omega(maxj+1), gamma(maxj+1);
    R = P = x; X.setZero(); omega.setZero(); gamma.setOnes();
    int j = 0; bool cnvg = false;
    while(j < maxj - 1){
        if(j == 0) gamma(j) = 1.0 / delta(j);
        else gamma(j) = 1.0 / (delta(j) - omega(j-1) / gamma(j-1));
        omega(j) = beta(j+1) * gamma(j) * beta(j+1) * gamma(j);
        rho(j+1) = -beta(j+1) * gamma(j) * rho(j);
        // CG vectors update
        X += gamma(j) * P;
        R = rho(j+1) * U.col(j+1);
        P = R + omega(j) * P;
        j++;
        res_norm = R.norm();
        //cout << "Error at step " << j << " is " << res_norm << endl;
        if(res_norm <= tol){
          cnvg = true;
          //break;
        }
    }
   // if(cnvg) cout << "Converged after " << j << " iterations." << endl;
   // else cout << "Failed to converge after " << j << " iterations." << endl;
    return X;
}

VectorXf aAplusbIinvx(Lseed seed, float setvg, float varcmp, float ve, VectorXf x){
    float setve = 1.0 - setvg;
    float sigma = ve / varcmp * setvg - setve;
    return lanczos_solve(seed, x, sigma) * setvg / varcmp;
}

VectorXf remlewithlanczos(VectorXf ytemp, int numc, int timemethod){
    clock_t start, end;
    start = clock();
    VectorXf temp = Hec(ytemp,numc);
    float varcmp = temp(0);
    float a = (n - 1.0)/(n - numc), b = - temp(1) / (n - numc);
    cout << "the initial value of variance given by HE is " << varcmp << endl;
    end = clock();
    cout <<  "HE using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    
    start = clock();
    VectorXf Viy(n);
    Lseed aa = lanczos_seed(0.5, ytemp);
    float yViy = 0.0, d = 0.0, ve = 0.0;
    cout << aa.j << endl;
    for (int loop = 0; loop < 8; loop++){
        ve = a + b * varcmp;
        Viy = aAplusbIinvx(aa, 0.5, varcmp, ve, ytemp);
        yViy = ytemp.dot(Viy);
        d = Viy.dot(Actimesx(Viy,timemethod) + b * Viy);
        varcmp += (yViy - n + numc) / d;
        cout << varcmp <<endl;
        if( abs((yViy - n + numc) / d ) < 0.00001) break;
    }
    Eigen::Vector2f varcmps(varcmp, a + b * varcmp);
    end = clock();
    cout <<  "remle is over, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return varcmps;
}



VectorXf remlesavememory(VectorXf ytemp, int numc, int timemethod){
    clock_t start, end;
    start = clock();
    VectorXf temp = Hec(ytemp,numc);
    float varcmp = temp(0);
    float a = (n - 1.0)/(n - numc), b = - temp(1) / (n - numc);
    cout << "the initial value of variance given by HE is " << varcmp << endl;
    end = clock();
    cout <<  "HE using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    start = clock();
    VectorXf Viy(n);
    float yViy = 0.0, d = 0.0, ve = 0.0;
    for (int loop = 0; loop < 5; loop++){
        ve = a + b * varcmp;
        Viy = conjugate(varcmp, ve, ytemp, timemethod);
        yViy = ytemp.dot(Viy);
        d = Viy.dot(Actimesx(Viy,timemethod) + b * Viy);
        varcmp += (yViy - n + numc) / d;
        cout << varcmp << endl;
        if(std::abs((yViy - n + numc) / d) < 0.000001) break;
    }
    Eigen::Vector2f varcmps(varcmp, a + b * varcmp);
    end = clock();
    cout <<  "remle is over, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return varcmps;
}

float Rademacher(float dummy) {
  static default_random_engine engine(std::time(0));
  static std::uniform_real_distribution<float> uni_dist(-1, 1);
  return (uni_dist(engine) > 0) ? 1.0 : -1.0;
}

float calcutrace(float vg, float ve, int timemethod, int numofrt){
    //20230505
    float r = 0.0;
    VectorXf x(n),Vix(n);
    for(int i = 0; i < numofrt; i++){
       // x = VectorXf::Zero(n).unaryExpr([](float dummy){return rnorm(e);});
        //x = VectorXf::Zero(n).unaryExpr(ptr_fun(Rademacher));
        x = VectorXf::Zero(n).unaryExpr([](float val) { return Rademacher(val); });
        Vix = conjugate(vg, ve, x, timemethod);
        r += Vix.dot(x);
    }
    return r / numofrt;
}

float calcutracewithlanczos(MatrixXf xs, float maxvg, float vg, float ve, vector<Lseed> seed, int numofrt, int numofy){
    //20230505
    float r = 0.0;
    for(int i = numofy; i < numofrt + numofy; i++){
        //r += aAplusbIinvx(seed[i], maxvg, vg, ve, xs.col(i)).dot(xs.col(i));
        r += aAplusbIinvx(seed[i], maxvg, vg, ve, seed[i].U.col(0)).dot(seed[i].U.col(0)) * n;
    }
    return r / numofrt;
}




VectorXf remlerandtrwithlanczos(VectorXf ytemp, int numc, float maxvg, int numofrt){
    float r = 0.0;
    clock_t start, end;
    start = clock();
    VectorXf varcmps = Hec(ytemp,numc), R(2);
    varcmps(1) = 1.0 - varcmps(0);
    cout << "the initial value of variance given by HE is " << varcmps(0) << endl;
    end = clock();
    cout <<  "HE using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    
    start = clock();
    VectorXf Viy(n), ViViy(n), AViy(n), ViAViy(n);
    MatrixXf H(2,2), Hi(2,2);
    MatrixXf xs = MatrixXf::Zero(n, numofrt + 1).unaryExpr([](float val) { return Rademacher(val); });
    xs.col(0) = ytemp;
    vector<Lseed> seed(numofrt + 1);
    for (int i = 0; i < numofrt + 1; i++){
        seed[i] = lanczos_seed(maxvg, xs.col(i));
        cout << "seed " << i << ": " << seed[i].j << endl;
    }
    
    for (int loop = 0; loop < 5; loop++){
        Viy = aAplusbIinvx(seed[0], maxvg, varcmps(0),varcmps(1), ytemp);
        AViy = Actimesx(Viy, 1);
        ViAViy = conjugate(varcmps(0),varcmps(1), AViy, 1);
        H(0,0) = AViy.dot(ViAViy);
        H(0,1) = H(1,0) = (Viy.dot(AViy) - varcmps(0) * H(0,0)) / varcmps(1);
        ViViy = (Viy - ViAViy * varcmps(0)) / varcmps(1);
        H(1,1) = ViViy.dot(Viy);
        Hi << H(1,1), - H(0,1), - H(0,1), H(0,0);
        Hi /= H(0,0) * H(1,1) - H(0,1) * H(0,1);
        r = calcutracewithlanczos(xs, maxvg, varcmps(0),varcmps(1), seed, numofrt, 1);
        R(0) = (n - r * varcmps(1))/varcmps(0); R(1) = r;
        //cout << r << endl;
        R(0) -= Viy.dot(AViy);
        R(1) -= (Viy.dot(Viy) + C / varcmps(1));
        varcmps -= Hi * R;
        cout << varcmps.transpose() << endl;
        if(varcmps(0) > maxvg) {varcmps(0) = maxvg; varcmps(1) = 1.0 - maxvg;}
    }
    end = clock();
    cout <<  "REML using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    return varcmps;
}

VectorXf remlerandtr(VectorXf ytemp, int numc,  int timemethod){
    //float varcmp = 1.5*Remlone(gtemp, ytemp);
    //cout << varcmp << endl;
    //float varcmp = Hecal(gtemp, ytemp);
    float vg = 0.5, ve = 1.0 - vg, r = 0.0;
   //vg = HeAB(ytemp), ve = 1 - vg;
    VectorXf varcmps(2), R(2);
    varcmps << vg, ve;
    cout << "the initial value of variance given by HE is " << vg << endl;
    
    VectorXf Viy(n), ViViy(n), AViy(n), ViAViy(n);
    MatrixXf H(2,2), Hi(2,2);
    for (int loop = 0; loop < 10; loop++){
        Viy = conjugate(varcmps(0),varcmps(1), ytemp, timemethod);
        AViy = Actimesx(Viy, timemethod);
        ViAViy = conjugate(varcmps(0),varcmps(1), AViy, timemethod);
        H(0,0) = AViy.dot(ViAViy);
        H(0,1) = H(1,0) = (Viy.dot(AViy) - varcmps(0) * H(0,0)) / varcmps(1);
        ViViy = (Viy - ViAViy * varcmps(0)) / varcmps(1);
        H(1,1) = ViViy.dot(Viy);
        Hi << H(1,1), - H(0,1), - H(0,1), H(0,0);
        Hi /= H(0,0) * H(1,1) - H(0,1) * H(0,1);
        r = calcutrace(varcmps(0),varcmps(1),timemethod, 10);
        R(0) = (n - r * varcmps(1))/varcmps(0); R(1) = r;
        cout << R << endl;
        R(0) -= Viy.dot(AViy);
        R(1) -= (Viy.dot(Viy) + C / varcmps(1));
        varcmps -= Hi * R;
        cout << varcmps.transpose() << endl;
    }
    return varcmps;
}


//float calcuf(vector<Lseed> seed, VectorXf ytemp, float maxvg, float delta){
//    float r = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0;
//    VectorXf Hiy(n);
//    Hiy = aAplusbIinvx(seed[0], maxvg, 1.0, delta, ytemp);
//    yHy = ytemp.dot(Hiy);
//    yHHy = Hiy.dot(Hiy);
//    yHAHy = yHy - delta * yHHy;
//    trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, seed, numofrt);
//    trAHi = n - delta * trHi;
//    //f = (trAHi/yHAHy)/(trHi/yHHy);
//    f = trAHi * yHy / yHAHy / (n - numc);
//}

VectorXf remlregwithlanczos(VectorXf ytemp, int numc, float maxvg, int numofrt){
    float delta = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0;
    clock_t start, end;
    start = clock();
    VectorXf Hiy(n);
    MatrixXf xs = MatrixXf::Zero(n, numofrt + 1).unaryExpr([](float val) { return Rademacher(val); });
    xs.col(0) = ytemp;
    vector<Lseed> seed(numofrt + 1);
    for (int i = 0; i < numofrt + 1; i++){
        seed[i] = lanczos_seed(maxvg, xs.col(i));
        cout << "seed " << i << ": " << seed[i].j << endl;
    }
    end = clock();
    cout <<  "Seeding time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    
//    ofstream out("/Users/tim/Desktop/seedtest.txt");
//    if (!out.is_open()) cout << "can not open the file\n";
//
//
//    for (int i = 0; i < numofrt + 1; i++){
//        out << setprecision(8) << seed[i].U << endl;
//        out << setprecision(8) << seed[i].beta.transpose()<< endl;
//        out << setprecision(8) << seed[i].delta.transpose()<< endl;
//    }
//    out.close();
    
    
    start = clock();
    float deltabig = 1.0 / maxvg - 1.0, deltasmall = 5.0 / maxvg - 1.0; //lower bound of h2 is h2/5
    for (int loop = 0; loop < 30; loop++){
        if(loop == 0) delta = deltabig;
        if(loop == 1) delta = deltasmall;
        Hiy = aAplusbIinvx(seed[0], maxvg, 1.0, delta, ytemp);
        //cout << Hiy.head(10).transpose() <<endl;
        yHy = ytemp.dot(Hiy);
        yHHy = Hiy.dot(Hiy);
        yHAHy = yHy - delta * yHHy;
        trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, seed, numofrt, 1);
        trAHi = n - delta * trHi;
        //f = (trAHi/yHAHy)/(trHi/yHHy);
        f = trAHi * yHy / yHAHy / (n - numc);
        cout << 1.0 / (delta + 1.0) << " " << f - 1.0 << endl;
        if (abs(f - 1.0) < 1e-9) break;
        else{
            if (f > 1.0) deltabig = delta;
            if (f < 1.0) deltasmall = delta;
            delta = (deltabig + deltasmall) / 2.0;
        }
        
    }
    end = clock();
    cout <<  "REMLreg time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    
    {
        float r = 0.0;
        clock_t start, end;
        start = clock();
        VectorXf varcmps = Hec(ytemp,numc), R(2);
        varcmps(1) = 1.0 - varcmps(0);
        cout << "the initial value of variance given by HE is " << varcmps(0) << endl;
        end = clock();
        cout <<  "HE time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
        
        start = clock();
        VectorXf Viy(n), ViViy(n), AViy(n), ViAViy(n);
        MatrixXf H(2,2), Hi(2,2);
        for (int loop = 0; loop < 10; loop++){
            Viy = aAplusbIinvx(seed[0], maxvg, varcmps(0),varcmps(1), ytemp);
            AViy = Actimesx(Viy, 1);
            ViAViy = conjugate(varcmps(0),varcmps(1), AViy, 1);
            H(0,0) = AViy.dot(ViAViy);
            H(0,1) = H(1,0) = (Viy.dot(AViy) - varcmps(0) * H(0,0)) / varcmps(1);
            ViViy = (Viy - ViAViy * varcmps(0)) / varcmps(1);
            H(1,1) = ViViy.dot(Viy);
            Hi << H(1,1), - H(0,1), - H(0,1), H(0,0);
            Hi /= H(0,0) * H(1,1) - H(0,1) * H(0,1);
            r = calcutracewithlanczos(xs, maxvg, varcmps(0),varcmps(1), seed, numofrt, 1);
            R(0) = (n - r * varcmps(1))/varcmps(0); R(1) = r;
            //cout << r << endl;
            R(0) -= Viy.dot(AViy);
            R(1) -= (Viy.dot(Viy) + C / varcmps(1));
            varcmps -= Hi * R;
            cout << varcmps(0)/varcmps.sum() << endl;
            if(varcmps(0) > maxvg) {varcmps(0) = maxvg; varcmps(1) = 1.0 - maxvg;}
        }
        end = clock();
        cout <<  "REML time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }

    VectorXf h2_vg(2);
    h2_vg << 1.0 / (delta + 1.0) , yHy / (n - numc);
    return h2_vg;
}




float trAc1Ac2(int i, int j){
    float xxxxAA, term1, term4;
    term1 = (float)(_singlebin[i].A.cast<double>().cwiseProduct(_singlebin[j].A.cast<double>()).sum() * 2 +
     _singlebin[i].B.cast<double>().cwiseProduct(_singlebin[j].B.cast<double>()).sum() * 2 +
                    _singlebin[i].diag.cast<double>().cwiseProduct(_singlebin[j].diag.cast<double>()).sum());
    //xxxxA = (_singlebin[i].CmatA.cwiseProduct(_XXCmat)).sum();
    xxxxAA = (_singlebin[i].CmatA * _singlebin[j].CmatA.transpose()).cwiseProduct(_XXdinv.cast<float>()).sum();
    MatrixXf xxxA1x = _singlebin[i].CmatA * _XXCmat.transpose();
    MatrixXf xxxA2x = _singlebin[j].CmatA * _XXCmat.transpose();
    term4 = (xxxA1x * xxxA2x).diagonal().sum();
    return term1 - xxxxAA * 2 + term4;
}

void read_grmlist(string phefile, string grmlist,  string covfile){
    clock_t start, end;
    read_realphe_all(phefile);
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
        cout << grmitem << endl;
    } while (!list.eof());
    list.close();
    r = grms.size();
    read_grmid(grms[0]);
    _y = merge_pheid_grmid();
    cout <<  n << " indviduals are in common" <<endl;
    read_realcov_ofcommonid(covfile);
    cout << _y.head(5).transpose() << endl;
    _y = proj_x(_y);
    _y -= VectorXf::Constant(n, _y.mean());
    _y /= (_y.norm() / sqrt(n - 1));
   
    _landmark.clear();
    _landmark.reserve(n);
    //_J.clear();
     for (int i = 0; i< n; i++){
        //_J.push_back(i);
        _landmark.push_back((long long)(grmloc[i]) * ( (long long)(grmloc[i]) + 1) / 2);
     }
   
    
    _singlebin.resize(r);
    VectorXf Ay(n);
    for (int i = 0; i < r; i++) {
        read_grmABmiss(grms[i] + ".grm.bin");
        //_singlebin[i].remle = remlewithlanczos(_y,C,1);
        //_singlebin[i].remlreg = remlregwithlanczos(_y, C, 0.25, 30);  //the largest h2 of all bins
        _CmatA.resize(C,n);  // x'A
        for (int j = 0; j < C; j++){
            _CmatA.row(j) = grmABtimesvector(_Cmat.col(j));
        }
        _singlebin[i].A = _A;
        _singlebin[i].B = _B;
        _singlebin[i].diag = _diag;
        _singlebin[i].CmatA = _CmatA;
        Ay = proj_x(grmABtimesvector(_y));
        _singlebin[i].yAy = Ay.dot(_y);
    }
    
    _C.resize(r + 1, r + 1);
    start = clock();
    for (int i = 0; i < (r + 1); i++) {
        for (int j = 0; j <= i; j++) {
            if(i == r)
            {
                if(j == r) _C(r, r) = n - C;
                else  _C(r, j) = _C(j, r) = _singlebin[j].diag.sum() - (_singlebin[j].CmatA.cwiseProduct(_XXCmat)).sum();
            }
            else
                _C(i, j) = _C(j, i) = trAc1Ac2(i, j);
        }
    }
    MatrixXf S = _C;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < i; j++) {
            S(i, j) = S(j, i) = _C(i, r) * _C(j, r) / (n - C);
        }
    }
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1);
    MatrixXd Cd = _C.cast<double>();
    MatrixXd Ci = Cd.ldlt().solve(Imatd);
    MatrixXd Sd = S.cast<double>();
    _StoC = (Cd.ldlt().solve(Sd)).cast<float>();
    end = clock();
  //  cout <<  "StoC matrix is ready, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << "the StoC matrix is:"<< endl << _StoC << endl;

//
//    VectorXf remlelist(r + 1), remllist(r + 1), right(r + 1);
//    for (int i = 0; i < r; i++) {
//        remlelist(i) = _singlebin[i].remle(0);
//        remllist(i) = _singlebin[i].remlreg(1);
//        right(i) = _singlebin[i].yAy;
//    }
//    remlelist(r) = (1.0 - remlelist.head(r).sum()) / n * (n - C);
//    remllist(r) = (1.0 - remllist.head(r).sum()) / n * (n - C);
//    right(r) =  n - 1.0;
//    remlestc = _StoC * remlelist;
//    remlstc = _StoC * remllist;
//      mhe = (Ci * right.cast<double>()).cast<float>();
//    cout << mhe.transpose() << endl;

}

VectorXf multiActimesxc(Grmbin &bin, VectorXf xtemp){
    int halfn;
    VectorXf x = proj_x(xtemp);
    if(n % 2 == 0){
        halfn = n/2;
        VectorXf y1 = x.head(halfn);
        VectorXf y2 = x.tail(halfn);
        VectorXf Vy1 = bin.A.selfadjointView<Eigen::Lower>() * y1 + bin.B.transpose() * y2;
        VectorXf Vy2 = bin.A.selfadjointView<Eigen::Upper>() * y2 + bin.B * y1;
        VectorXf Vy(n);
        Vy << Vy1, Vy2;
        return proj_x(Vy + bin.diag.cwiseProduct(x));
    }
    else{
        halfn = (n + 1)/2;
        VectorXf xpan(n + 1);
        xpan << x, 0;
        VectorXf y1 = xpan.head(halfn);
        VectorXf y2 = xpan.tail(halfn);
        VectorXf Vy1 = bin.A.selfadjointView<Eigen::Lower>() * y1 + bin.B.transpose() * y2;
        VectorXf Vy2 = bin.A.selfadjointView<Eigen::Upper>() * y2 + bin.B * y1;
        VectorXf Vy(n + 1);
        Vy << Vy1, Vy2;
        return proj_x(Vy.head(n) + bin.diag.cwiseProduct(x));
    }
}

VectorXf calcuR1(float ve, vector<MatrixXf> Arx){
    VectorXf R1(r + 1);
    int numofrt = Arx[0].cols();
    MatrixXf Vix(n, numofrt);
    for (int i = 0; i < numofrt; i++) {
        Vix.col(i) = conjugate(1.0, ve, Arx[r].col(i), 1);
    }
    for (int i = 0; i < r + 1; i++) {
        R1(i) = Arx[i].cwiseProduct(Vix).sum();
    }
    return R1 / numofrt;
}

VectorXf mremlrandtr(VectorXf ytemp, int numc, int numofrt, int timemethod){
    //float varcmp = 1.5*Remlone(gtemp, ytemp);
    clock_t start, end;
    VectorXf varcmp = mhe;
    VectorXf R1(r + 1), R2(r + 1);
    cout << "the initial value of variance given by HE is " << mhe.transpose() << endl;
    
    VectorXf Viy(n), ViViy(n);
    MatrixXf AI(r + 1, r + 1), AViy(n, r + 1), ViAViy(n, r + 1);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), AId, AIi;
    MatrixXf xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });
    vector<MatrixXf> Arx(r + 1);
    for (int i = 0; i < r ; i++) {
        Arx[i].resize(n, numofrt);
        for (int j = 0; j < numofrt; j++) {
            Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j));
        }
    }
    Arx[r] = xs;
    for (int loop = 0; loop < 6; loop++){
        start = clock();
        _A.setZero();_B.setZero();_diag.setZero();
        for (int i = 0; i < r; i++) {
            _A += _singlebin[i].A * varcmp(i);
            _B += _singlebin[i].B * varcmp(i);
            _diag += _singlebin[i].diag * varcmp(i);
        }
        Viy = conjugate(1.0, varcmp(r), ytemp, timemethod);
       
        for (int i = 0; i < r; i++) {
            AViy.col(i) = multiActimesxc(_singlebin[i], Viy);
            R2(i) = Viy.dot(AViy.col(i));
            ViAViy.col(i) = conjugate(1.0, varcmp(r), AViy.col(i), timemethod);
        }
        AViy.col(r) = Viy;
        R2(r) = Viy.dot(Viy) + C / varcmp(r);
        ViAViy.col(r) = conjugate(1.0, varcmp(r), Viy, timemethod);
        for (int i = 0; i < r + 1; i++) {
            for (int j = 0; j <= i; j++) {
                AI(i, j) = AI(j, i) = AViy.col(i).cwiseProduct(ViAViy.col(j)).sum();
            }
        }
       
        R1 = calcuR1(varcmp(r), Arx);
        AId = AI.cast<double>();
        AIi = AId.ldlt().solve(Imatd);
        varcmp -= AIi.cast<float>() * (R1 - R2);
        //cout << R1 << endl;
        //cout << R2 << endl;
        //cout << AI << endl;
        cout << varcmp.transpose() << endl;
        if(loop == 0){
            end = clock();
            cout <<  "1 iteration using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
        }
           
    }
    return varcmp;
}


//20240114
void read_realcov_nomiss(string covfile){
    _Cmat.resize(n, C);
    //MatrixXf cov = MatrixXf::Zero(n, C);
    string index,covitem,s;
    int i,j,temp,indextemp;
    ifstream covin(covfile, ios::in );
    if (!covin.is_open()) cout << "can not open the covfile\n";
    for (i = 0; i < n; i++) {
        indextemp = grmid[i]; //the first number in each line of grmfile: id
        do {
            getline(covin, covitem);
            temp = stoi(covitem);
        } while (indextemp != temp);
        istringstream is(covitem);
        is >> s; is >> s;
        for(j = 0; j < C; j++){
            is >> s;
            _Cmat(i,j) = stof(s);
        }
        _Cmat(i,C - 1) = 1.0;
    }
    covin.close();
    //MatrixXf XX = _Cmat.transpose() * _Cmat;
    MatrixXd XXd = _Cmat.cast<double>().transpose() * _Cmat.cast<double>();
    MatrixXd Imatd = MatrixXd::Identity(C,C);
    //MatrixXd XXd = XX.cast<double>();
    _XXdinv = XXd.ldlt().solve(Imatd);
    //_XXCmat = _XXdinv.cast<float>() * _Cmat.transpose();
    _XXCmat = (_XXdinv * _Cmat.transpose().cast<double>()).cast<float>();
}



float remlregwithpreLanzcos(VectorXf ytemp, int numc, float maxvg, int numofrt, string seedfile){
    float delta = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0;
    clock_t start, end;
    start = clock();
    VectorXf Hiy(n);
    MatrixXf xs = MatrixXf::Zero(n, numofrt + 1).unaryExpr([](float val) { return Rademacher(val); });
    xs.col(0) = ytemp;
    vector<Lseed> preseed(numofrt + 1);
    MatrixXf Utemp(n, 11);
    VectorXf btemp(11);
    ifstream in(seedfile);
    if (!in.is_open()) cout << "can not open the file\n";
    for (int i = 0; i < numofrt + 1; i++){
        for (int j = 0; j < Utemp.rows(); j++) {
            for (int k = 0; k < Utemp.cols(); k++) {
                in >> Utemp(j, k);
            }
        }
        for (int j = 0; j < btemp.size(); j++) {
            in >> btemp(j);
        }
        preseed[i].U = Utemp;
        preseed[i].beta = btemp;
        for (int j = 0; j < btemp.size(); j++) {
            in >> btemp(j);
        }
        preseed[i].delta = btemp;
        preseed[i].j = 12;
    }
    in.close();
    end = clock();
    cout <<  "Seed reading time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << preseed[0].beta.transpose() << endl;
    //cout << preseed[50].delta.transpose() << endl;
    //cout << preseed[0].U.row(0) << endl;
    //cout << preseed[0].U.row(1) << endl;
    //cout << preseed[0].U(0,0) - 0.030039<< endl;

    start = clock();
    float deltabig = 1.0 / maxvg - 1.0, deltasmall = 5.0 / maxvg - 1.0;
    for (int loop = 0; loop < 30; loop++){
        if(loop == 0) delta = deltabig;
        if(loop == 1) delta = deltasmall;
        //Hiy = aAplusbIinvx(preseed[0], maxvg, 1.0, delta, ytemp);
        //yHy = ytemp.dot(Hiy);
        //yHHy = Hiy.dot(Hiy);
        //20240114 modify the following 3 codes
        Hiy = aAplusbIinvx(preseed[0], maxvg, 1.0, delta, preseed[0].U.col(0));
        yHy = preseed[0].U.col(0).dot(Hiy) * (n - 1);
        yHHy = Hiy.dot(Hiy) * (n - 1);
        yHAHy = yHy - delta * yHHy;
        trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, preseed, numofrt, 1);
        trAHi = n - delta * trHi;
        //f = (trAHi/yHAHy)/(trHi/yHHy);
        f = trAHi * yHy / yHAHy / (n - numc);
        //cout << 1.0 / (delta + 1.0) << " " << abs(f - 1.0) << endl;
        if (abs(f - 1.0) < 1e-9) break;
        else{
            if (f > 1.0) deltabig = delta;
            if (f < 1.0) deltasmall = delta;
            delta = (deltabig + deltasmall) / 2.0;
        }
        
    }
    end = clock();
    cout <<  "REMLreg time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    return 1.0 / (delta + 1.0);
}



//20240114 add
MatrixXf read_mulphe(string phefile, int numofy){
    MatrixXf ymat = MatrixXf::Zero(n, numofy);
    string index,pheitem,s;
    int i,j,temp,indextemp;
    ifstream phe(phefile, ios::in );
    if (!phe.is_open()) cout << "can not open the file phe\n";
    for (i = 0; i < n; i++) {
        getline(phe,pheitem);
        istringstream is(pheitem);
        is >> s; is >> s;
        for (j = 0; j < numofy; j++) {
            is >> s;
            ymat(i, j) = stof(s);
        }
    }
    phe.close();
    
    VectorXf yytemp = VectorXf::Zero(n);
    for (j = 0; j < numofy; j++) {
        yytemp = ymat.col(j);
        yytemp -= VectorXf::Constant(n, yytemp.mean());
        yytemp /= (yytemp.norm() / sqrt(n - 1));
        ymat.col(j) = yytemp;
    }
    return ymat;
}

//20240114
void ymatproj(MatrixXf& ymat){
    VectorXf yt(n);
    int numofy = ymat.cols();
    int sqrt_n = std::sqrt(n - 1);
    for (int i = 0; i < numofy; i++) {
        yt = proj_x(ymat.col(i));
        yt -= VectorXf::Constant(n, yt.mean());
        yt /= (yt.norm() / sqrt_n);
        ymat.col(i) = yt;
    }
}

void writeseeds(MatrixXf ymat, float maxvg, int numofrt, int numofy, string seedfile){
    int totalseeds = numofrt + numofy;
    MatrixXf xs = MatrixXf::Zero(n, totalseeds).unaryExpr([](float val) { return Rademacher(val); });
    xs.leftCols(numofy) = ymat.leftCols(numofy);
    vector<Lseed> seed(totalseeds);
    for (int i = 0; i < totalseeds; i++){
        seed[i] = lanczos_seed(maxvg, xs.col(i));
        cout << "seed " << i << ": " << seed[i].j << endl;
    }
    
    ofstream out(seedfile);
    if (!out.is_open()) cout << "can not open the file\n";
    
    
    for (int i = 0; i < totalseeds; i++){
        out << std::setprecision(8) << seed[i].U << endl;
        out << std::setprecision(8) << seed[i].beta.transpose()<< endl;
        out << std::setprecision(8) << seed[i].delta.transpose()<< endl;
    }
    out.close();
}

vector<Lseed> readseeds(float maxvg, int numofrt, int numofy, string seedfile){
    int totalseeds = numofrt + numofy;
    vector<Lseed> preseed(numofrt + totalseeds);
    MatrixXf Utemp(n, 19);
    VectorXf btemp(19);
    cout << "ok" << endl;

    ifstream in(seedfile);
    if (!in.is_open()) cout << "can not open the file\n";
    for (int i = 0; i < totalseeds; i++){
        for (int j = 0; j < Utemp.rows(); j++) {
            for (int k = 0; k < Utemp.cols(); k++) {
                in >> Utemp(j, k);
            }
        }
        for (int j = 0; j < btemp.size(); j++) {
            in >> btemp(j);
        }
        preseed[i].U = Utemp;
        preseed[i].beta = btemp;
        for (int j = 0; j < btemp.size(); j++) {
            in >> btemp(j);
        }
        preseed[i].delta = btemp;
        preseed[i].j = 19;
    }
    in.close();
    return preseed;
}

VectorXf remlreg_seeds(MatrixXf ymat, int numc, float maxvg, int numofrt, int numofy, string seedfile){
    float delta = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0, deltabig = 0.0, deltasmall = 0.0;
    VectorXf h2s(numofy);
    clock_t start, end;
    start = clock();
    VectorXf Hiy(n);
    MatrixXf xs = MatrixXf::Zero(n, 2);
    vector<Lseed> preseed = readseeds(maxvg, numofrt, numofy, seedfile);
    end = clock();
    cout <<  "Seed reading time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << preseed[0].beta.transpose() << endl;
    //cout << preseed[50].delta.transpose() << endl;
    //cout << preseed[0].U.row(0) << endl;
    //cout << preseed[0].U.row(1) << endl;
    //cout << preseed[0].U(0,0) - 0.030039<< endl;

    start = clock();
    for (int i = 0; i < numofy; i++){
        deltabig = 1.0 / maxvg - 1.0; deltasmall = 200.0 / maxvg - 1.0;
        for (int loop = 0; loop < 30; loop++){
            if(loop == 0) delta = deltabig;
            if(loop == 1) delta = deltasmall;
            //Hiy = aAplusbIinvx(preseed[0], maxvg, 1.0, delta, ytemp);
            //yHy = ytemp.dot(Hiy);
            //yHHy = Hiy.dot(Hiy);
            //20240114 modify the following 3 codes
            Hiy = aAplusbIinvx(preseed[i], maxvg, 1.0, delta, preseed[i].U.col(0));
            yHy = preseed[i].U.col(0).dot(Hiy) * (n - 1);
            yHHy = Hiy.dot(Hiy) * (n - 1);
            yHAHy = yHy - delta * yHHy;
            trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, preseed, numofrt, numofy);
            trAHi = n - delta * trHi;
            //f = (trAHi/yHAHy)/(trHi/yHHy);
            f = trAHi * yHy / yHAHy / (n - numc);
            cout << 1.0 / (delta + 1.0) << " " << f - 1.0 << endl;
            if (abs(f - 1.0) < 1e-9) break;
            else{
                if (f > 1.0) deltabig = delta;
                if (f < 1.0) deltasmall = delta;
                delta = (deltabig + deltasmall) / 2.0;
            }
        }
        h2s(i) = 1.0 / (delta + 1.0);
    }
    end = clock();
    cout <<  "REMLreg time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    return h2s;
}

VectorXf testformulLanzcos(MatrixXf ymat, int numc, float maxvg, int numofrt, int numofy, string seedfile){
    VectorXf h2s(numofy);
    int totalseeds = numofrt + numofy;
    float delta = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0, deltabig = 0.0, deltasmall = 0.0;
    VectorXf Hiy(n);
    MatrixXf xs = MatrixXf::Zero(n, totalseeds).unaryExpr([](float val) { return Rademacher(val); });
    xs.leftCols(numofy) = ymat.leftCols(numofy);
    vector<Lseed> seed(totalseeds);
    for (int i = 0; i < totalseeds; i++){
        seed[i] = lanczos_seed(maxvg, xs.col(i));
        cout << "seed " << i << ": " << seed[i].j << endl;
    }
    
//    ofstream out(seedfile);
//    if (!out.is_open()) cout << "can not open the file\n";
//
//
//    for (int i = 0; i < totalseeds; i++){
//        out << std::setprecision(10) << seed[i].U << endl;
//        out << std::setprecision(10) << seed[i].beta.transpose()<< endl;
//        out << std::setprecision(10) << seed[i].delta.transpose()<< endl;
//    }
//    out.close();
    
    for (int i = 0; i < numofy; i++){
        deltabig = 1.0 / maxvg - 1.0; deltasmall = 500.0 / maxvg - 1.0;
        for (int loop = 0; loop < 30; loop++){
            if(loop == 0) delta = deltabig;
            if(loop == 1) delta = deltasmall;
            Hiy = aAplusbIinvx(seed[i], maxvg, 1.0, delta, seed[i].U.col(0));
            yHy = seed[i].U.col(0).dot(Hiy) * (n - 1);
            yHHy = Hiy.dot(Hiy) * (n - 1);
            yHAHy = yHy - delta * yHHy;
            trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, seed, numofrt, numofy);
            trAHi = n - delta * trHi;
            //f = (trAHi/yHAHy)/(trHi/yHHy);
            f = trAHi * yHy / yHAHy / (n - numc);
             cout << 1.0 / (delta + 1.0) << " " << f - 1.0 << endl;
            if (abs(f - 1.0) < 1e-9) break;
            else{
                if (f > 1.0) deltabig = delta;
                if (f < 1.0) deltasmall = delta;
                delta = (deltabig + deltasmall) / 2.0;
            }
            
        }
        h2s(i) = 1.0 / (delta + 1.0);
    }
    
    {
        for (int i = 0; i < numofy; i++){
            float r = 0.0;
            VectorXf ytemp = ymat.col(i);
            VectorXf varcmps = Hec(ytemp,numc), R(2);
            varcmps(1) = 1.0 - varcmps(0);
            cout << "the initial value of variance given by HE is " << varcmps(0) << endl;
            VectorXf Viy(n), ViViy(n), AViy(n), ViAViy(n);
            MatrixXf H(2,2), Hi(2,2);
            for (int loop = 0; loop < 5; loop++){
                Viy = aAplusbIinvx(seed[i], maxvg, varcmps(0),varcmps(1), ytemp);
                AViy = Actimesx(Viy, 1);
                ViAViy = conjugate(varcmps(0),varcmps(1), AViy, 1);
                H(0,0) = AViy.dot(ViAViy);
                H(0,1) = H(1,0) = (Viy.dot(AViy) - varcmps(0) * H(0,0)) / varcmps(1);
                ViViy = (Viy - ViAViy * varcmps(0)) / varcmps(1);
                H(1,1) = ViViy.dot(Viy);
                Hi << H(1,1), - H(0,1), - H(0,1), H(0,0);
                Hi /= H(0,0) * H(1,1) - H(0,1) * H(0,1);
                r = calcutracewithlanczos(xs, maxvg, varcmps(0),varcmps(1), seed, numofrt, numofy);
                R(0) = (n - r * varcmps(1))/varcmps(0); R(1) = r;
                //cout << r << endl;
                R(0) -= Viy.dot(AViy);
                R(1) -= (Viy.dot(Viy) + C / varcmps(1));
                varcmps -= Hi * R;
                cout << varcmps(0)/varcmps.sum() << endl;
                if(varcmps(0) > maxvg) {varcmps(0) = maxvg; varcmps(1) = 1.0 - maxvg;}
            }

        }
    }

    return h2s;
}


//20240116
Lseed lanczos_multiseeds(Grmbin &bin, float varcmp, VectorXf x){
    float tol = 5e-2, ve = 1.0 - varcmp, res_norm = 1.0; //its just enough for 50k
    int j = 0, maxit = 50;
    bool cnvg = false;
    MatrixXf U(n,maxit);
    VectorXf R(n),Vv(n), beta(maxit), rho(maxit), omega(maxit), delta(maxit), gamma(maxit);
    U.setZero(); Vv.setZero();R.setZero(); beta.setZero(); rho.setZero(); omega.setZero(); delta.setZero(); gamma.setOnes();

    // Initial values
    beta(0) = rho(0) = x.norm();
    U.col(0) = x / beta(0); //v1 = arbitrary norm1 vector

    // while(j < maxit){
    while(j < maxit - 1){   //20240114
    // Lanczos iteration
        Vv = varcmp * multiActimesxc(bin, U.col(j)) + ve * U.col(j);
        //Vv = Vtimesx(varcmp, ve, U.col(j));
        delta(j) = U.col(j).dot(Vv);
        if(j == 0) Vv = Vv - delta(j) * U.col(j);
        else Vv = Vv - delta(j) * U.col(j) - beta(j) * U.col(j-1);
        beta(j+1) = Vv.norm();
        U.col(j+1) = Vv / beta(j+1);

        // CG coefficients update
        if(j == 0) gamma(j) = 1.0 / delta(j);
        else gamma(j) = 1.0 / (delta(j) - omega(j-1) / gamma(j-1));
        omega(j) = beta(j+1) * gamma(j) * beta(j+1) * gamma(j);
        rho(j+1) = -beta(j+1) * gamma(j) * rho(j);

        // CG vectors update
        R = rho(j+1) * U.col(j+1);
        j++;
        res_norm = R.norm();
        //cout << "Error at step " << j << " is " << res_norm << endl;
        if(res_norm <= tol && j > 30){
            cnvg = true;
            break;
        }
    }
    // if(cnvg) cout << "Converged after " << j << " iterations." << endl;
    // else cout << "Failed to converge after " << j << " iterations." << endl;

    Vv = Vtimesx(varcmp, ve, U.col(j));
    delta(j) = U.col(j).dot(Vv);
    
    Lseed seed;
    seed.j = j; seed.U = U.leftCols(j); seed.beta = beta.head(j); seed.delta = delta.head(j); seed.rho = rho.head(j);
    return seed;
}

    
    
    
    
    
vector<vector<Lseed>> makemseeds(MatrixXf ymat, MatrixXf xs, float maxvg, int numofrt, int numofy, string seedfile){
    int totalseeds = numofrt + numofy;
    vector<vector<Lseed>> mseeds(r, vector<Lseed>(totalseeds));
    for (int i = 0; i < r; i++){
        for(int j = 0; j < totalseeds; j++){
            mseeds[i][j] = lanczos_multiseeds(_singlebin[i], maxvg, xs.col(j));
            cout << "seed " << j << " of group" << i << ": " << mseeds[i][j].j << endl;
        }
    }
    //cout << mseeds[1][0].U.col(1).head(5).transpose() << endl;
    return mseeds;
}

MatrixXf calcuViX(float maxvg, float vg, float ve, vector<Lseed> seed, int numofrt, int numofy){
    MatrixXf ViX(n, numofrt);
    for( int i = 0; i < numofrt; i++){
        ViX.col(i) = aAplusbIinvx(seed[i + numofy], maxvg, vg, ve, seed[i + numofy].U.col(0)) * sqrt(n);
    }
   // return r / numofrt;
    return ViX;
}


VectorXf multiLanzcos(MatrixXf ymat, int numc, float maxvg, int numofrt, int numofy, string seedfile){
    int totalseeds = numofrt + numofy;
    VectorXf varcmp = mhe;
    VectorXf R1(r + 1), R2(r + 1);
    cout << "the initial value of variance given by HE is " << mhe.transpose() << endl;
    
    
    VectorXf Viy(n), right(r + 1);
    MatrixXf Left(r + 1, r + 1), ViX(n, numofrt);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), Leftd, Lefti;
    MatrixXf xs = MatrixXf::Zero(n, totalseeds).unaryExpr([](float val) { return Rademacher(val); });
    xs.leftCols(numofy) = ymat.leftCols(numofy);  //y1,y2...yn,x1,x2,...xn
    
    vector<MatrixXf> Arx(r + 1);
    for (int i = 0; i < r ; i++) {
        Arx[i].resize(n, numofrt);
        for (int j = 0; j < numofrt; j++) {
            Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j + numofy));
        }
    }
    Arx[r] = xs.rightCols(numofrt);
    
    vector<vector<Lseed>> mseeds = makemseeds(ymat, xs, maxvg, numofrt, numofy, seedfile);
    
    for (int j = 0; j < r ; j++){
        Left(r, j) = _singlebin[j].diag.sum() - (_singlebin[j].CmatA.cwiseProduct(_XXCmat)).sum(); //tr(Ac)
    }
    Left(r, r) = n - C;
    right(r) = n - 1;
    
    for (int loop = 0; loop < 30; loop++){
        for (int i = 0; i < r ; i++) {
            Viy = aAplusbIinvx(mseeds[i][0], maxvg, varcmp(i), varcmp(r), ymat.col(0));
            ViX = calcuViX(maxvg, varcmp(i), varcmp(r), mseeds[i], numofrt, numofy);
            right(i) = Viy.dot(ymat.col(0));
            for (int j = 0; j < (r + 1); j++){
                Left(i, j) = Arx[j].cwiseProduct(ViX).sum()/numofrt - C;
            }
        }
        Leftd = Left.cast<double>();
       // Lefti = Leftd.ldlt().solve(Imatd);  // Left is not positive
       // Lefti = Leftd.lu().inverse();
        Lefti = Leftd.fullPivHouseholderQr().solve(Imatd);  //both are ok under MatrixXd
        varcmp = Lefti.cast<float>() * right;
//        cout << Left << endl;
//        cout << right << endl;
//        cout << Lefti << endl;
        cout << varcmp.transpose() << endl;
    }
    
    return varcmp;
}

void writemultiseeds(MatrixXf ymat, float maxvg, int numofrt, int numofy, string multiseedfile){
    int totalseeds = numofrt + numofy;
    MatrixXf xs = MatrixXf::Zero(n, totalseeds).unaryExpr([](float val) { return Rademacher(val); });
    xs.leftCols(numofy) = ymat.leftCols(numofy);  //y1,y2...yn,x1,x2,...xn

    vector<vector<Lseed>> mseeds = makemseeds(ymat, xs, maxvg, numofrt, numofy, multiseedfile);
    
    vector<MatrixXf> Arx(r + 1);
    for (int i = 0; i < r ; i++) {
        Arx[i].resize(n, numofrt);
        for (int j = 0; j < numofrt; j++) {
            Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j + numofy));
        }
    }
    Arx[r] = xs.rightCols(numofrt);
    
    ofstream out(multiseedfile);
    if (!out.is_open()) cout << "can not open the file\n";
    
    for (int i = 0; i < r ; i++) {
        for(int j = 0; j < totalseeds; j++){
            out << mseeds[i][j].j << endl;
        }
    }
    for (int i = 0; i < r ; i++) {
        for(int j = 0; j < totalseeds; j++){
            out << setprecision(10) << mseeds[i][j].U << endl;
            out << setprecision(10) << mseeds[i][j].beta.transpose()<< endl;
            out << setprecision(10) << mseeds[i][j].delta.transpose()<< endl;
        }
    }
    for (int i = 0; i < (r + 1) ; i++) {
        out << setprecision(10) << Arx[i] << endl;
    }
    for (int i = 0; i < r ; i++){
        out << setprecision(10) << _singlebin[i].diag.sum() - (_singlebin[i].CmatA.cwiseProduct(_XXCmat)).sum() << endl; //tr(Ac)
    }
    out.close();
}

VectorXf read_multiLanzcos(MatrixXf ymat, int numc, float maxvg, int numofrt, int numofy, string multiseedfile){
    int totalseeds = numofrt + numofy;
    n = 46567;
    r = 5;
    mhe.resize(r + 1);
    mhe << 0.05, 0.05,0.05,0.05,0.05,0.75;
    VectorXf varcmp = mhe;
    VectorXf R1(r + 1), R2(r + 1);
    cout << "the initial value of variance given by HE is " << mhe.transpose() << endl;
    
    
    VectorXf Viy(n), right(r + 1);
    MatrixXf Left(r + 1, r + 1), ViX(n, numofrt);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), Leftd, Lefti;
    //MatrixXf xs(n, totalseeds);
   // xs.leftCols(numofy) = ymat.leftCols(numofy);  //y1,y2...yn,x1,x2,...xn
    
    vector<MatrixXf> Arx(r + 1);
    
    vector<vector<Lseed>> mseeds(r, vector<Lseed>(totalseeds));
    
    MatrixXf Utemp;
    VectorXf btemp, deltatemp;

    ifstream in(multiseedfile);
    if (!in.is_open()) cout << "can not open the file\n";
    
    for (int i = 0; i < r ; i++) {
        for(int j = 0; j < totalseeds; j++){
            in >> mseeds[i][j].j;
            cout << mseeds[i][j].j << endl;
        }
    }
    for (int i = 0; i < r ; i++) {
        for (int j = 0; j < totalseeds; j++){
            Utemp.resize(n, mseeds[i][j].j);
            btemp.resize(mseeds[i][j].j);
            deltatemp.resize(mseeds[i][j].j);
            for (int k = 0; k < Utemp.rows(); k++) {
                for (int l = 0; l < Utemp.cols(); l++) {
                    in >> Utemp(k, l);
                }
            }
            for (int k = 0; k < btemp.size(); k++) {
                in >> btemp(k);
            }
            for (int k = 0; k < deltatemp.size(); k++) {
                in >> deltatemp(k);
            }
            mseeds[i][j].U = Utemp;
            mseeds[i][j].beta = btemp;
            mseeds[i][j].delta = deltatemp;
        }
    }
    for (int i = 0; i < (r + 1) ; i++) {
        Arx[i].resize(n, numofrt);
        for (int k = 0; k < n; k++) {
            for (int l = 0; l < numofrt; l++) {
                in >> Arx[i](k, l);
            }
        }
    }
    for (int i = 0; i < r ; i++){
        in >> Left(r, i);
    }
    in.close();
    Left(r, r) = n - C;
    right(r) = n - 1;
    
    //cout << mseeds[0][0].U.col(0).tail(5).transpose() << endl;
    for (int t = 0; t < numofy; t++){
        cout << "trait "<< t << " : " << endl;
        varcmp = mhe;
        for (int loop = 0; loop < 30; loop++){
            for (int i = 0; i < r ; i++) {
                Viy = aAplusbIinvx(mseeds[i][t], maxvg, varcmp(i), varcmp(r), ymat.col(t));
                // cout << Viy.head(5).transpose() << endl;
                ViX = calcuViX(maxvg, varcmp(i), varcmp(r), mseeds[i], numofrt, numofy);
                right(i) = Viy.dot(ymat.col(t));
                for (int j = 0; j < (r + 1); j++){
                    Left(i, j) = Arx[j].cwiseProduct(ViX).sum()/numofrt - C;
                }
            }
            Leftd = Left.cast<double>();
            // Lefti = Leftd.ldlt().solve(Imatd);  // Left is not positive
            // Lefti = Leftd.lu().inverse();
            Lefti = Leftd.fullPivHouseholderQr().solve(Imatd);  //both are ok under MatrixXd
            varcmp = Lefti.cast<float>() * right;
            // cout << Left << endl;
            // cout << right << endl;
            //        cout << Lefti << endl;
            cout << varcmp.transpose() << endl;
        }
    }
    
    return varcmp/varcmp.sum();
}
    
    
VectorXf testmulti_seedsforsingle(MatrixXf ymat, int numc, float maxvg, int numofrt, int numofy, string multiseedfile){
    n = 46567;r = 5;
    float delta = 0.0, f = 0.0, trAHi = 0.0, trHi = 0.0, yHAHy = 0.0, yHHy = 0.0, yHy = 0.0, deltabig = 0.0, deltasmall = 0.0;
    VectorXf h2s(numofy);
    VectorXf Hiy(n);
    MatrixXf xs = MatrixXf::Zero(n, 2);
    vector<vector<Lseed>> mseeds(r, vector<Lseed>(numofrt + numofy));
    
    MatrixXf Utemp;
    VectorXf btemp, deltatemp;

    ifstream in(multiseedfile);
    if (!in.is_open()) cout << "can not open the file\n";
    
    for (int i = 0; i < r ; i++) {
        for(int j = 0; j < numofrt + numofy; j++){
            in >> mseeds[i][j].j;
           // cout << mseeds[i][j].j << endl;
        }
    }
    for (int i = 0; i < 1 ; i++) {
        for (int j = 0; j < numofrt + numofy; j++){
            Utemp.resize(n, mseeds[i][j].j);
            btemp.resize(mseeds[i][j].j);
            deltatemp.resize(mseeds[i][j].j);
            for (int k = 0; k < Utemp.rows(); k++) {
                for (int l = 0; l < Utemp.cols(); l++) {
                    in >> Utemp(k, l);
                }
            }
            for (int k = 0; k < btemp.size(); k++) {
                in >> btemp(k);
            }
            for (int k = 0; k < deltatemp.size(); k++) {
                in >> deltatemp(k);
            }
            mseeds[i][j].U = Utemp;
            mseeds[i][j].beta = btemp;
            mseeds[i][j].delta = deltatemp;
        }
    }

    for (int i = 0; i < numofy; i++){
        cout << "trait "<< i << " : " << endl;
        deltabig = 1.0 / maxvg - 1.0; deltasmall = 5000.0 / maxvg - 1.0;
        for (int loop = 0; loop < 30; loop++){
            if(loop == 0) delta = deltabig;
            if(loop == 1) delta = deltasmall;
            //Hiy = aAplusbIinvx(preseed[0], maxvg, 1.0, delta, ytemp);
            //yHy = ytemp.dot(Hiy);
            //yHHy = Hiy.dot(Hiy);
            //20240114 modify the following 3 codes
            Hiy = aAplusbIinvx(mseeds[0][i], maxvg, 1.0, delta, mseeds[0][i].U.col(0));  //only test the 1st group
            yHy = mseeds[0][i].U.col(0).dot(Hiy) * (n - 1);
            yHHy = Hiy.dot(Hiy) * (n - 1);
            yHAHy = yHy - delta * yHHy;
            trHi = calcutracewithlanczos(xs, maxvg, 1.0, delta, mseeds[0], numofrt, numofy);
            trAHi = n - delta * trHi;
            //f = (trAHi/yHAHy)/(trHi/yHHy);
            f = trAHi * yHy / yHAHy / (n - numc);
            cout << 1.0 / (delta + 1.0) << " " << f - 1.0 << endl;
            if (abs(f - 1.0) < 1e-9) break;
            else{
                if (f > 1.0) deltabig = delta;
                if (f < 1.0) deltasmall = delta;
                delta = (deltabig + deltasmall) / 2.0;
            }
        }
        h2s(i) = 1.0 / (delta + 1.0);
    }

    return h2s;
}
    
void read_grmAB_faster(string file){
    int halfn = (n + 1)/2;
    _A.setZero(halfn, halfn);
    _B.setZero(halfn, halfn);
    _diag.setZero(n);
    int i = 0, j = 0;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;

    for(i = 0; i< halfn; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        for(j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else _A(i,j) = f_buf;
        }
    }

    for(i = halfn; i< n; i++){
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else if(j < halfn) _B(i - halfn,j) = f_buf;
            else _A(j - halfn, i - halfn) = f_buf;
        }
    }
    fin.close();
}

//20240428
void read_grmA_oneCPU(string file, int start, int end){
    long long loc = (long long)start * (long long)(start + 1) / 2;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.seekg(loc * (long long)size, ios::beg);
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        for(int j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else _A(i,j) = f_buf;
        }
    }
    fin.close();
}

void read_grmAB_oneCPU(string file, int start, int end){
    int halfn = (n + 1)/2;
    long long loc = (long long)start * (long long)(start + 1) / 2;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.seekg(loc * (long long)size, ios::beg);
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(int j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) = f_buf;
            else if(j < halfn) _B(i - halfn,j) = f_buf;
            else _A(j - halfn, i - halfn) = f_buf;
        }
    }
    fin.close();
}

struct Covariates {
    int rows = -1;
    // Total columns of the covariates file.
    int cols = -1;
    std::vector<int> ids;
    // Flattened data matrix, it's eigen friendly.
    std::vector<float> data;

    // We ignored the first two columns, so data size should be rows * (cols - 2)
    bool valid() { return rows > 0 && cols > 0 && data.size() == rows * (cols - 2); }
};

bool is_blank(const std::string &s) {
    return std::all_of(s.begin(), s.end(),
                       [](unsigned char c) { return std::isspace(c); });
}

// C++ std::stof is slower than its C version.
bool fast_strtof(const std::string &s, float &out) {
    char *end = nullptr;
    const char *cstr = s.c_str();
    out = std::strtof(cstr, &end);
    return end != cstr && *end == '\0';
}

/// Read the covariates file, ignore the blank lines.
Covariates read_covariates(const std::string &cov_file) {
    Covariates cov;

    std::ifstream ifs_cov(cov_file);

    if (!ifs_cov.is_open()) {
        std::cout << "open cov file failed, path=" << cov_file << std::endl;
        return cov;
    }

    std::string line;
    std::getline(ifs_cov, line);
    std::istringstream iss(line);
    std::string token;
    int cols = 0;

    while (iss >> token) {
        cols += 1;
    }

    if (cols == 0) {
        std::cout << "open cov file success, but data has empty columns"
                  << std::endl;
        return cov;
    }

    // Reset file stream
    ifs_cov.clear();
    ifs_cov.seekg(0, std::ios::beg);

    // Pre-reserve can decrease peak RSS memory.
    cov.data.reserve(13000000);
    cov.ids.reserve(500000);

    int rows = 0;
    while (std::getline(ifs_cov, line)) {
        if (is_blank(line)) {
            continue;
        }

        rows += 1;

        iss.clear();
        iss.str(line);
        iss >> token;
        cov.ids.push_back(std::stoi(token));

        // Ignore the first two columns
        iss >> token;

        while (iss >> token) {
            float val = 0.0f;
            if (fast_strtof(token, val)) {
                cov.data.push_back(val);
            } else {
                cov.data.push_back(0.0f);
            }
        }
    }

    cov.rows = rows;
    cov.cols = cols;

    if (!cov.valid()) {
        std::cout << "read cov success, but cov data is empty" << std::endl;
        return cov;
    }

    return cov;
}

void read_realcov_search_withmiss_v2(string covfile) {
    PerfTimer _perf_timer("read_realcov_search_withmiss_v2");
    Covariates cov = read_covariates(covfile);
    C = cov.cols - 1;

    spdlog::info("Reading quantitative covariates from [{}], n={}, rows={}, cols={}", covfile, n, cov.rows, cov.cols);

    Eigen::Map<Eigen::MatrixXf> Cmat502492(cov.data.data(), cov.rows,
                                           cov.cols - 2);
    Eigen::Map<Eigen::VectorXi> idcov =
        Eigen::VectorXi::Map(cov.ids.data(), cov.ids.size());

    _Cmat.resize(n, C);

    for (int i = 0; i < n; i++) {
        // the first number in each line of grmfile: id
        int grm_idx = grmid[nomissgrmid[i]];
        for (int j = 0; j < cov.rows; j++) {
            if (idcov(j) == grm_idx) {
                _Cmat.row(i).head(C - 1) = Cmat502492.row(j);
                break;
            }
        }
        _Cmat(i, C - 1) = 1.0;
    }

    // NOTE: idcov and Cmat502492 are cleared here!
    cov.data.clear();
    cov.data.shrink_to_fit();
    cov.ids.clear();
    cov.ids.shrink_to_fit();

    spdlog::info("{} covariates of {},  individuals were read\n", C - 1, cov.rows);
    // MatrixXf XX = _Cmat.transpose() * _Cmat;
    MatrixXd XXd = _Cmat.cast<double>().transpose() * _Cmat.cast<double>();
    MatrixXd Imatd = MatrixXd::Identity(C, C);
    // MatrixXd XXd = XX.cast<double>();
    _XXdinv = XXd.ldlt().solve(Imatd);
    //_XXCmat = _XXdinv.cast<float>() * _Cmat.transpose();
    _XXCmat = (_XXdinv * _Cmat.transpose().cast<double>()).cast<float>();
}

//20240620
void read_realcov_search_withmiss(string covfile){
    PerfTimer _perf_timer("read_realcov_search_withmiss");
    int linenum = countValidLines(covfile);
    C = countItemnumber(covfile) - 1;
    cout << "Reading quantitative covariates from [" << covfile << "]" << endl;
    MatrixXf Cmat502492(linenum, C - 1);
    VectorXi idcov(linenum);
    _Cmat.resize(n, C);
    //MatrixXf cov = MatrixXf::Zero(n, C);
    string index,covitem,s;
    int i,j,temp,indextemp;
    
    ifstream covin(covfile, ios::in);
    if (!covin.is_open()) cout << "can not open the covfile\n";
    for (i = 0; i < linenum; i++) {
        getline(covin, covitem);
        temp = stoi(covitem);
        idcov(i) = temp;
        istringstream is(covitem);
        is >> s; is >> s;
        for(j = 0; j < C - 1; j++){
            is >> s;
            try{
                Cmat502492(i,j) = stof(s);
            } catch(const std::invalid_argument& e){break;}
        }
    }
    for (i = 0; i < n; i++) {
        indextemp = grmid[nomissgrmid[i]]; //the first number in each line of grmfile: id
        for(j = 0; j< linenum; j++){
            if(idcov(j) == indextemp){
                _Cmat.row(i).head(C - 1) = Cmat502492.row(j);
                break;
            }
        }
        _Cmat(i,C - 1) = 1.0;
    }
    //cout << _Cmat.bottomRightCorner(5, 5) << endl;
    covin.close();
    Cmat502492.resize(0, 0);
    cout << C - 1 << " covariates of " << linenum << " individuals were read.\n" << endl;
    //MatrixXf XX = _Cmat.transpose() * _Cmat;
    MatrixXd XXd = _Cmat.cast<double>().transpose() * _Cmat.cast<double>();
    MatrixXd Imatd = MatrixXd::Identity(C,C);
    //MatrixXd XXd = XX.cast<double>();
    _XXdinv = XXd.ldlt().solve(Imatd);
    //_XXCmat = _XXdinv.cast<float>() * _Cmat.transpose();
    _XXCmat = (_XXdinv * _Cmat.transpose().cast<double>()).cast<float>();
}



//20240620
void read_grmA_oneCPU_withmiss(const std::string& file, int start, int end){
    long long loclast = (long long)(nomissgrmid[start]) * (long long)(nomissgrmid[start] + 1) / 2;
    long long locnow, locdif, loctemp;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.seekg(loclast * (long long)size, ios::beg);
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) cout << "Reading id:" << i << endl;
        for(int j = 0; j<= i; j++){
            loctemp = (long long)(nomissgrmid[i]) * (long long)(nomissgrmid[i] + 1) / 2 + (long long)nomissgrmid[j] + (long long)1;
            locdif = loctemp - loclast - (long long)1;
            //if(i<4) cout << nomissgrmid[i] << endl;
            if(locdif > 0)
                fin.seekg(locdif * (long long)size, ios::cur);
            fin.read((char*) &f_buf, size);
            //if(i<4) cout << f_buf << endl;
            if(i == j) _diag(i) = f_buf;
            else _A(i,j) = f_buf;
            loclast = loctemp;
        }
    }
    fin.close();
}

//20240622
void read_grmA_oneCPU_withmiss_batch(const string& grm_path, int64_t start, int64_t end) {
    std::ifstream fin(grm_path, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);

        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]];
            if(UNLIKELY(i == j)) _diag(i) = val;
            else _A(i,j) = val;
        }
    }
    fin.close();
}





//20240622
void read_grmAB_oneCPU_withmiss_batch(const string& file, int64_t start, int64_t end) {
    int64_t halfn = (n + 1) / 2;
    std::ifstream fin(file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";

    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i < end; i++) {
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t loclast = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(loclast * sizeof(float), ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++){
            float val = values[nomissgrmid[j]];
            if(UNLIKELY(i == j)) _diag(i) = val;
            else if(j < halfn) _B(i - halfn,j) = val;
            else _A(j - halfn, i - halfn) = val;
        }
    }
    fin.close();
}


//20240620
void read_grmA_oneCPU_forrt_withmiss(const string& grm_path, int64_t start, int64_t end, float var){
    std::ifstream fin(grm_path, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i < end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]] * var;
            if(UNLIKELY(i == j)) _diag(i) += val;
            else _A(i, j) += val;
        }
    }
    fin.close();
}

//20240620
void read_grmAB_oneCPU_forrt_withmiss(const string& grm_path, int64_t start, int64_t end, float var) {
    ifstream fin(grm_path, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";

    // Increase internal buffer
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    fin.rdbuf()->pubsetbuf(stream_buffer.data(), stream_buffer.size());

    int64_t halfn = (n + 1)/2;
    std::vector<char> buffer((nomissgrmid[end - 1] + 1) * sizeof(float));
    for(int i = start; i < end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        int64_t nomiss_i = nomissgrmid[i];
        int64_t offset = nomiss_i * (nomiss_i + 1) / 2;
        fin.seekg(offset * sizeof(float), ios::beg);
        fin.read(buffer.data(), sizeof(float) * (nomiss_i + 1));
        float* values = reinterpret_cast<float*>(buffer.data());
        for(int j = 0; j <= i; j++) {
            float val = values[nomissgrmid[j]] * var;
            if(UNLIKELY(i == j)) _diag(i) += val;
            else if(j < halfn) _B(i - halfn,j) += val;
            else _A(j - halfn, i - halfn) += val;
        }
    }
    fin.close();
}


//20240428 20240623mis
void read_grmAB_faster_parallel(const std::string& file){
    PerfTimer _perf_timer(__FUNCTION__);

    int64_t halfn = (n + 1) / 2;
    _A.setZero(halfn, halfn);
    _B.setZero(halfn, halfn);
    _diag.setZero(n);

    int chunks = std::min(16, omp_get_num_procs());
    int64_t upper_count = halfn * (halfn + 1) / 2;
    int64_t chunk_size = upper_count / chunks;
    MatrixXi chunk_ranges(2, chunks);
    chunk_ranges(0, 0) = 0;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * chunk_size * i - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = halfn;
    
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        //read_grmA_oneCPU(file, startend(0, i), startend(1, i));
        //read_grmA_oneCPU_withmiss(file, startend(0, i), startend(1, i));
        read_grmA_oneCPU_withmiss_batch(file, chunk_ranges(0, i), chunk_ranges(1, i));
    }

    int64_t lower_count = n * (n + 1) / 2 - upper_count;
    chunk_size = lower_count / chunks;
    chunk_ranges(0, 0) = halfn;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * (chunk_size * i + upper_count)  - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = n;
    
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        //read_grmAB_oneCPU(file, startend(0, i), startend(1, i));
        //read_grmAB_oneCPU_withmiss(file, startend(0, i), startend(1, i));
        read_grmAB_oneCPU_withmiss_batch(file, chunk_ranges(0, i), chunk_ranges(1, i));
    }
    
}

void testremlemhe(string phefile, string grmlist,  string covfile, MatrixXf ymat, int numc, int numofy){
    clock_t start, end;
    
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
        cout << grmitem << endl;
    } while (!list.eof());
    list.close();
    r = grms.size();
    //read_grmid(grms[0]);
    cout <<  n << " indviduals are in common" <<endl;
    cout << "There are "<< r << " GRM groups" <<endl;
    //read_realcov_ofcommonid(covfile);
    
    MatrixXf remles(numofy, r + 1), rights(numofy, r + 1);
    _singlebin.resize(r);
    VectorXf Ay(n);
    for (int i = 0; i < r; i++) {
        start = clock();
        read_grmAB_faster(grms[i] + ".grm.bin");
        end = clock();
        cout <<  "the time for reading grm" << i + 1 << " is: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
        for(int t = 0; t < numofy; t++){
            //remles(t,i) = remlewithlanczos(ymat.col(t),C,1)(0);
            //_singlebin[i].remlreg = remlregwithlanczos(_y, C, 0.25, 20);  //the largest h2 of all bins
            _CmatA.resize(C,n);  // x'A
            for (int j = 0; j < C; j++){
                _CmatA.row(j) = grmABtimesvector(_Cmat.col(j));
            }
            _singlebin[i].A = _A;
            _singlebin[i].B = _B;
            _singlebin[i].diag = _diag;
            _singlebin[i].CmatA = _CmatA;
            Ay = proj_x(grmABtimesvector(ymat.col(t)));
            //_singlebin[i].yAy = Ay.dot(_y);
            rights(t,i) = Ay.dot(ymat.col(t));
        }
    }
    for(int t = 0; t < numofy; t++) {
        //remles(t,r) = (1.0 - remles.row(t).head(r).sum()) / n * (n - C);
        rights(t,r) = n - 1.0;
    }
    //cout << remles << endl;
    
    _C.resize(r + 1, r + 1);
    start = clock();
    for (int i = 0; i < (r + 1); i++) {
        for (int j = 0; j <= i; j++) {
            if(i == r)
            {
                if(j == r) _C(r, r) = n - C;
                else  _C(r, j) = _C(j, r) = _singlebin[j].diag.sum() - (_singlebin[j].CmatA.cwiseProduct(_XXCmat)).sum();
            }
            else
                _C(i, j) = _C(j, i) = trAc1Ac2(i, j);
        }
    }
    cout << _C << endl;
    MatrixXf S = _C;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < i; j++) {
            S(i, j) = S(j, i) = _C(i, r) * _C(j, r) / (n - C);
        }
    }
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1);
    MatrixXd Cd = _C.cast<double>();
    MatrixXd Ci = Cd.ldlt().solve(Imatd);
    MatrixXd Sd = S.cast<double>();
    _StoC = (Cd.ldlt().solve(Sd)).cast<float>();
    end = clock();
    cout <<  "StoC matrix is ready, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "the StoC matrix is:" << endl << _StoC << endl;
    
    cout << "the remlestc is:" << endl;
    cout << (_StoC * remles.transpose()).transpose() << endl;
    cout << "the mhe is:" << endl;
    mhes = (Ci * rights.cast<double>().transpose()).cast<float>().transpose();
    cout << mhes << endl;

//    VectorXf right(r + 1);
//    for (int i = 0; i < r; i++) {
//        remlelist(i) = _singlebin[i].remle(0);
//        remllist(i) = _singlebin[i].remlreg(1);
//        right(i) = _singlebin[i].yAy;
//    }
   // remlelist(r) = (1.0 - remlelist.head(r).sum()) / n * (n - C);
    //remllist(r) = (1.0 - remllist.head(r).sum()) / n * (n - C);
    //right(r) =  n - 1.0;
    //remlestc = _StoC * remlelist;
    //remlstc = _StoC * remllist;
    //mhe = (Ci * right.cast<double>()).cast<float>();

}

MatrixXf mremlrandtr_mtraits(MatrixXf ymat, int numc, int numofrt, int timemethod){
    //float varcmp = 1.5*Remlone(gtemp, ytemp);
    clock_t start, end;
    MatrixXf mtres(ymat.cols(), r + 1);
    VectorXf R1(r + 1), R2(r + 1);
    VectorXf Viy(n), ViViy(n);
    MatrixXf AI(r + 1, r + 1), AViy(n, r + 1), ViAViy(n, r + 1);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), AId, AIi;
    MatrixXf xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });
    vector<MatrixXf> Arx(r + 1);
    for (int i = 0; i < r ; i++) {
        Arx[i].resize(n, numofrt);
        for (int j = 0; j < numofrt; j++) {
            Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j));
        }
    }
    Arx[r] = xs;
    for(int t = 0; t< ymat.cols(); t++){
        xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });
        for (int i = 0; i < r ; i++) {
            Arx[i].resize(n, numofrt);
            for (int j = 0; j < numofrt; j++) {
                Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j));
            }
        }
        Arx[r] = xs;
        VectorXf varcmp = mhes.row(t);
        for (int loop = 0; loop < 6; loop++){
            start = clock();
            _A.setZero();_B.setZero();_diag.setZero();
            for (int i = 0; i < r; i++) {
                _A += _singlebin[i].A * varcmp(i);
                _B += _singlebin[i].B * varcmp(i);
                _diag += _singlebin[i].diag * varcmp(i);
            }
            if(loop == 0) _check = true;
            Viy = conjugate(1.0, varcmp(r), ymat.col(t), timemethod);
            _check = false;
           
            for (int i = 0; i < r; i++) {
                AViy.col(i) = multiActimesxc(_singlebin[i], Viy);
                R2(i) = Viy.dot(AViy.col(i));
                ViAViy.col(i) = conjugate(1.0, varcmp(r), AViy.col(i), timemethod);
            }
            AViy.col(r) = Viy;
            R2(r) = Viy.dot(Viy) + C / varcmp(r);
            ViAViy.col(r) = conjugate(1.0, varcmp(r), Viy, timemethod);
            for (int i = 0; i < r + 1; i++) {
                for (int j = 0; j <= i; j++) {
                    AI(i, j) = AI(j, i) = AViy.col(i).cwiseProduct(ViAViy.col(j)).sum();
                }
            }
           
            R1 = calcuR1(varcmp(r), Arx);
            AId = AI.cast<double>();
            AIi = AId.ldlt().solve(Imatd);
            varcmp -= AIi.cast<float>() * (R1 - R2);
            //cout << R1 << endl;
            //cout << R2 << endl;
            //cout << AI << endl;
            cout << varcmp.transpose() << endl;
            if(loop == 0){
                end = clock();
                cout <<  "1 iteration using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
            }
        }
        mtres.row(t) = varcmp;
        //mtres.row(t) = varcmp/varcmp.sum();
    }
    return mtres;
}


//20240223
void read_realcov_search(string covfile){
    MatrixXf Cmat502492(502492, C - 1);
    VectorXi idcov(502492);
    _Cmat.resize(n, C);
    //MatrixXf cov = MatrixXf::Zero(n, C);
    string index,covitem,s;
    int i,j,temp,indextemp;
    ifstream covin(covfile, ios::in );
    if (!covin.is_open()) cout << "can not open the covfile\n";
    for (i = 0; i < 502492; i++) {
        getline(covin, covitem);
        temp = stoi(covitem);
        idcov(i) = temp;
        istringstream is(covitem);
        is >> s; is >> s;
        for(j = 0; j < C - 1; j++){
            is >> s;
            try{
                Cmat502492(i,j) = stof(s);
            } catch(const std::invalid_argument& e){break;}
        }
    }
    for (i = 0; i < n; i++) {
        indextemp = grmid[i]; //the first number in each line of grmfile: id
        for(j = 0; j< 502492; j++){
            if(idcov(j) == indextemp){
                _Cmat.row(i).head(C - 1) = Cmat502492.row(j);
                break;
            }
        }
        _Cmat(i,C - 1) = 1.0;
    }
    covin.close();
    Cmat502492.resize(0, 0);
    //MatrixXf XX = _Cmat.transpose() * _Cmat;
    MatrixXd XXd = _Cmat.cast<double>().transpose() * _Cmat.cast<double>();
    MatrixXd Imatd = MatrixXd::Identity(C,C);
    //MatrixXd XXd = XX.cast<double>();
    _XXdinv = XXd.ldlt().solve(Imatd);
    //_XXCmat = _XXdinv.cast<float>() * _Cmat.transpose();
    _XXCmat = (_XXdinv * _Cmat.transpose().cast<double>()).cast<float>();
}

//20240223
MatrixXf read_mulphe_search(string phefile, int numofy){
    int nraw = 349660;
    MatrixXf yraw = MatrixXf::Zero(nraw, numofy);
    VectorXi yid(nraw);
    MatrixXf ymat = MatrixXf::Zero(n, numofy);
    string index,pheitem,s;
    int i,j,temp,indextemp;
    ifstream phe(phefile, ios::in );
    if (!phe.is_open()) cout << "can not open the file phe\n";
    for (i = 0; i < nraw; i++) {
        getline(phe,pheitem);
        istringstream is(pheitem);
        is >> s; is >> s;
        yid(i) = stoi(s);
        for (j = 0; j < numofy; j++) {
            is >> s;
            yraw(i, j) = stof(s);
        }
    }
    for (i = 0; i < n; i++) {
        indextemp = grmid[i];
        for (j = 0; j < nraw; j++) {
            if(yid(j) == indextemp){
                ymat.row(i) = yraw.row(j);
                break;
            }
        }
    }
    phe.close();
    yraw.resize(0, 0);
    
    VectorXf yytemp = VectorXf::Zero(n);
    for (j = 0; j < numofy; j++) {
        yytemp = ymat.col(j);
        yytemp -= VectorXf::Constant(n, yytemp.mean());
        yytemp /= (yytemp.norm() / sqrt(n - 1));
        ymat.col(j) = yytemp;
    }
    return ymat;
}


//20240310
MatrixXf mremlrandtr_mtraits_small(MatrixXf ymat, int numc, int numofrt, int timemethod){
    //float varcmp = 1.5*Remlone(gtemp, ytemp);
    clock_t start, end;
    MatrixXf mtres(ymat.cols(), r + 1);
    VectorXf R1(r + 1), R2(r + 1);
    VectorXf Viy(n), ViViy(n);
    MatrixXf AI(r + 1, r + 1), AViy(n, r + 1), ViAViy(n, r + 1);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), AId, AIi;
    MatrixXf xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });
    vector<MatrixXf> Arx(r + 1);
    for (int i = 0; i < r ; i++) {
        Arx[i].resize(n, numofrt);
        for (int j = 0; j < numofrt; j++) {
            Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j));
        }
    }
    Arx[r] = xs;
    for(int t = 0; t< ymat.cols(); t++){
        xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });
        for (int i = 0; i < r ; i++) {
            Arx[i].resize(n, numofrt);
            for (int j = 0; j < numofrt; j++) {
                Arx[i].col(j) = multiActimesxc(_singlebin[i], xs.col(j));
            }
        }
        Arx[r] = xs;
        VectorXf varcmp = mhes.row(t);
        for (int loop = 0; loop < 6; loop++){
            start = clock();
            _A.setZero();_B.setZero();_diag.setZero();
            for (int i = 0; i < r; i++) {
                _A += _singlebin[i].A * varcmp(i);
                _B += _singlebin[i].B * varcmp(i);
                _diag += _singlebin[i].diag * varcmp(i);
            }
            if(loop == 0) _check = true;
            Viy = conjugate(1.0, varcmp(r), ymat.col(t), timemethod);
            _check = false;
           
            for (int i = 0; i < r; i++) {
                AViy.col(i) = multiActimesxc(_singlebin[i], Viy);
                R2(i) = Viy.dot(AViy.col(i));
                ViAViy.col(i) = conjugate(1.0, varcmp(r), AViy.col(i), timemethod);
            }
            AViy.col(r) = Viy;
            R2(r) = Viy.dot(Viy) + C / varcmp(r);
            ViAViy.col(r) = conjugate(1.0, varcmp(r), Viy, timemethod);
            for (int i = 0; i < r + 1; i++) {
                for (int j = 0; j <= i; j++) {
                    AI(i, j) = AI(j, i) = AViy.col(i).cwiseProduct(ViAViy.col(j)).sum();
                }
            }
           
            R1 = calcuR1(varcmp(r), Arx);
            AId = AI.cast<double>();
            AIi = AId.ldlt().solve(Imatd);
            varcmp -= AIi.cast<float>() * (R1 - R2);
            //cout << R1 << endl;
            //cout << R2 << endl;
            //cout << AI << endl;
            cout << varcmp.transpose() << endl;
            if(loop == 0){
                end = clock();
                cout <<  "1 iteration using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
            }
        }
        mtres.row(t) = varcmp;
        //mtres.row(t) = varcmp/varcmp.sum();
    }
    return mtres;
}

//20240517
VectorXf mhe_readtwiceform1(string grmfile, int groupid, long long count, long long unit){
    VectorXf grmseg(unit);
    int size = sizeof(float);
    float f_buf = 0.0;
    ifstream fin(grmfile, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    fin.seekg(count * (long long)size, ios::beg);
    for(long long j = 0; j < unit; j++){
        fin.read((char*) &f_buf, size);
        grmseg(j) = f_buf;
    }
    fin.close();
    return grmseg;
}

MatrixXf calcuterm1(string grmlist, int segment){
    clock_t start, end;
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
    } while (!list.eof());
    list.close();

    long long element = (long long) n * (long long)(n + 1) / 2;
    cout << element << endl;
    long long count = 0;
    long long unit = (element / (long long)segment) + 1;
    cout << unit << endl;
    MatrixXf grmseg(unit, r), temprr(r, r), rr(r, r);
    int size = sizeof(float);
    float f_buf = 0.0;
    rr.setZero();
    start = clock();
    for(int loop = 0; loop < segment; loop++){
        grmseg.setZero();
        //        for(int i = 0; i < r; i++){
//            ifstream fin(grms[i] + ".grm.bin", ios::in | ios::binary);
//            if (!fin.is_open()) cout << "can not open the file\n";
//            fin.seekg(count * (long long)size, ios::beg);
//            for(long long j = 0; j < unit; j++){
//                fin.read((char*) &f_buf, size);
//                grmseg(j, i) = f_buf;
//            }
//            fin.close();
//        }
#pragma omp parallel for
        for(int i = 0; i < r; i++){
            grmseg.col(i) = mhe_readtwiceform1(grms[i] + ".grm.bin", i, count, unit);
        }
        count += unit;
        if(loop == segment - 1) grmseg.bottomRows((int)((long long)unit * segment - element)).setZero();
        for(int i = 0; i < r; i++){
            for(int j = 0; j <= i; j++){
                temprr(i, j) = temprr(j, i) = (float)(grmseg.col(i).cast<double>().dot(grmseg.col(j).cast<double>()));
            }
        }
        rr += temprr;
        end = clock();
        cout <<  "Total time for "<< loop + 1 <<" loops is : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    return rr;
}

//20240623
VectorXf mhe_readtwiceform1_miss(string grmfile, int groupid, int start, int end){
    vector<float> grmsegm;
    long long loclast;
    ifstream fin( grmfile, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    for(int i = start; i< end; i++){
        loclast = (long long)(nomissgrmid[i]) * (long long)(nomissgrmid[i] + 1) / 2;
        fin.seekg(loclast * (long long)size, ios::beg);
        char buffer[size * (nomissgrmid[i] + 1)];
        fin.read(buffer, size * (nomissgrmid[i] + 1));
        float* values = reinterpret_cast<float*>(buffer);
        for(int j = 0; j<= i; j++){
            grmsegm.push_back(values[nomissgrmid[j]]);
        }
    }
    fin.close();
    VectorXf grmseg = VectorXf::Zero(grmsegm.size());
    grmseg = Eigen::Map<VectorXf> (&grmsegm[0], grmsegm.size());
    return grmseg;
}

//20240623
MatrixXf calcuterm1_miss(string grmlist, int segm){
    clock_t start, end;
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
    } while (!list.eof());
    list.close();
    
    long long element = (long long)n * (long long)(n + 1) / 2;
    long long unit = element / (long long)segm;
    MatrixXi startend(2, segm);
    startend(0, 0) = 0;
    for(int i = 1; i< segm; i++){
        startend(0, i) = floor(sqrt(2 * unit * (long long)i - 0.25) + 0.5);
    }
    startend.row(1).segment(0, segm - 1) = startend.row(0).segment(1, segm - 1);
    startend(1, segm - 1) = n;
    
    cout << startend << endl;
    MatrixXf grmseg(1, r), temprr(r, r), rr(r, r);
    rr.setZero();
    start = clock();
    long long loc1, loc2;
    for(int loop = 0; loop < segm; loop++){
        loc1 =(long long)startend(0, loop) * (long long)(startend(0, loop) + 1) / 2;
        loc2 =(long long)startend(1, loop) * (long long)(startend(1, loop) + 1) / 2;
        grmseg.resize(loc2 - loc1, r);
#pragma omp parallel for
        for(int i = 0; i < r; i++){
            grmseg.col(i) = mhe_readtwiceform1_miss(grms[i] + ".grm.bin", i, startend(0, loop), startend(1, loop));
        }
        for(int i = 0; i < r; i++){
            for(int j = 0; j <= i; j++){
                temprr(i, j) = temprr(j, i) = (float)(grmseg.col(i).cast<double>().dot(grmseg.col(j).cast<double>()));
            }
        }
        rr += temprr;
        end = clock();
       // cout <<  "Total time for "<< loop + 1 <<" loops is : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    return rr;
}

//20240408 20240428para 20240517for480G 20240623mis
void large_mhe_v3(string grmlist, string mhefile, MatrixXf ymat, int segment){
    int numofy = ymat.cols();
    clock_t start, end;
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
        cout << grmitem << endl;
    } while (!list.eof());
    list.close();
    r = grms.size();
    cout << "There are "<<  r <<" groups (except the error)"  << endl;
    
    struct mhebin{
        MatrixXf CmatA;
        VectorXf diag;
    };
    vector<mhebin> mhebins;
    mhebins.resize(r);
    MatrixXf rights(numofy, r + 1);
    VectorXf Ay(n);


    //apply another 240G
    // start = clock();
     int halfn = (n + 1)/2;
    // _singlebin.resize(1);
    // _singlebin[0].A.setZero(halfn, halfn); _singlebin[0].B.setZero(halfn, halfn);
    // end = clock();
    // cout <<  "Total time for apply 240G is : " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    start = clock();
    for (int i = 0; i < r; i++) {
        read_grmAB_faster_parallel(grms[i] + ".grm.bin");
        _CmatA.resize(C,n);  // x'A
        cout << "calculating the cov ";
#pragma omp parallel for
        for (int j = 0; j < C; j++){
            cout << j + 1 << " ";
            _CmatA.row(j) = grmABtimesvector(_Cmat.col(j));
        }
        cout << endl << "calculating the phenotype ";
        mhebins[i].diag = _diag;
        mhebins[i].CmatA = _CmatA;
#pragma omp parallel for
        for(int t = 0; t < numofy; t++){
            //remles(t,i) = remlewithlanczos(ymat.col(t),C,1)(0);
            cout << t + 1 << " ";
            //Ay = proj_x(grmABtimesvector(ymat.col(t)));
            //rights(t,i) = Ay.dot(ymat.col(t));
            rights(t,i) = proj_x(grmABtimesvector(ymat.col(t))).dot(ymat.col(t));
        }
        cout << endl;
        end = clock();
        cout <<  "Total time for "<<  i + 1 <<" groups is : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    for(int t = 0; t < numofy; t++) {
        //remles(t,r) = (1.0 - remles.row(t).head(r).sum()) / n * (n - C);
        rights(t,r) = n - 1.0;
    }
    _A.resize(1, 1); _B.resize(1, 1);
    //cout << remles << endl;
    cout << "OK1" << endl;
    float xxxxAA, term1, term4;
    MatrixXf rr1 = calcuterm1_miss(grmlist, segment);  //20240623
    MatrixXf rr2(r, r), rr3(r, r);
    cout << "OK2" << endl;
    for(int i = 0; i < r; i++){
        for(int j = 0; j <= i; j++){
            rr1(i, j) = rr1(j, i) = rr1(i, j) * 2.0 - (float)(mhebins[i].diag.cast<double>().dot(mhebins[j].diag.cast<double>()));
            rr2(i, j) = rr2(j, i) = (mhebins[i].CmatA * mhebins[j].CmatA.transpose()).cwiseProduct(_XXdinv.cast<float>()).sum();
            rr3(i, j) = rr3(j, i) = ((mhebins[i].CmatA * _XXCmat.transpose()) * (mhebins[j].CmatA * _XXCmat.transpose())).diagonal().sum();
        }
    }
    cout << "OK3" << endl;
    _C.resize(r + 1, r + 1);
    for (int j = 0; j < r; j++) {
        _C(r, j) = _C(j, r) = mhebins[j].diag.sum() - (mhebins[j].CmatA.cwiseProduct(_XXCmat)).sum();
    }
    _C.topLeftCorner(r, r) = rr1 - 2.0 * rr2 + rr3;
    _C(r, r) = n - C;
    cout << _C << endl;
    MatrixXf S = _C;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < i; j++) {
            S(i, j) = S(j, i) = _C(i, r) * _C(j, r) / (n - C);
        }
    }
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1);
    MatrixXd Cd = _C.cast<double>();
    MatrixXd Ci = Cd.ldlt().solve(Imatd);
    MatrixXd Sd = S.cast<double>();
    _StoC = (Cd.ldlt().solve(Sd)).cast<float>();
    //end = clock();
    //cout <<  "StoC matrix is ready, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "the StoC matrix is:" << endl << _StoC << endl;
    
    //cout << "the remlestc is:" << endl;
    //cout << (_StoC * remles.transpose()).transpose() << endl;
    cout << "the mhe is:" << endl;
    mhes = (Ci * rights.cast<double>().transpose()).cast<float>().transpose();
    cout << mhes << endl;
    
    ofstream out(mhefile);
    if (!out.is_open()) cout << "can not open the file\n";
    out << mhes << endl;
    out << setprecision(8) << (Cd.ldlt().solve(Sd)) << endl;
    out << setprecision(8) << Cd << endl;
    out.close();
}

//240408
void large_remle(string grmlist, string mhefile, MatrixXf ymat, int numofy){
    clock_t start, end;
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
        cout << grmitem << endl;
    } while (!list.eof());
    list.close();
    r = grms.size();
    cout << "There are "<<  r <<" groups (except the error)"  << endl;
    
    MatrixXf remles(numofy, r + 1);

    start = clock();
    for (int i = 0; i < r; i++) {
        read_grmAB_faster(grms[i] + ".grm.bin");
        cout << endl << "calculating the phenotype ";
        for(int t = 0; t < numofy; t++){
            remles(t,i) = remlewithlanczos(ymat.col(t),C,1)(0);
            cout << t + 1 << " ";
        }
        cout << endl;
        end = clock();
        cout <<  "Total time for "<<  i + 1 <<" groups is : " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    _A.resize(1, 1); _B.resize(1, 1);   //这里把读入的GRM清空防止后续算trace的时候增加内存！
    for (int t = 0; t < numofy; t++) {
        remles(t,r) = (1.0 - remles.row(t).head(r).sum()) / n * (n - C);
    }
    cout  << setprecision(8) << remles << endl;
}



//20240430
void read_grmA_oneCPU_forrt(string file, int start, int end, float var){
    long long loc = (long long)start * (long long)(start + 1) / 2;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.seekg(loc * (long long)size, ios::beg);
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        for(int j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) += f_buf * var;
            else _A(i,j) += f_buf * var;
        }
    }
    fin.close();
}

void read_grmAB_oneCPU_forrt(string file, int start, int end, float var){
    int halfn = (n + 1)/2;
    long long loc = (long long)start * (long long)(start + 1) / 2;
    ifstream fin( file, ios::in | ios::binary);
    if (!fin.is_open()) cout << "can not open the file\n";
    int size = sizeof (float);
    float f_buf = 0.0;
    fin.seekg(loc * (long long)size, ios::beg);
    for(int i = start; i< end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        for(int j = 0; j<= i; j++){
            fin.read((char*) &f_buf, size);
            if(i == j) _diag(i) += f_buf * var;
            else if(j < halfn) _B(i - halfn,j) += f_buf * var;
            else _A(j - halfn, i - halfn) += f_buf * var;
        }
    }
    fin.close();
}



//20240430
void read_grmAB_forrt_parallel(const std::string& file, float var){
    PerfTimer _perf_timer(__FUNCTION__);

    int64_t halfn = (n + 1) / 2;
    int chunks = std::min(16, omp_get_num_procs());
    int64_t upper_count = halfn * (halfn + 1) / 2;
    int64_t chunk_size = upper_count / chunks;
    MatrixXi chunk_ranges(2, chunks);

    // Compute grmA chunk range
    chunk_ranges(0, 0) = 0;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * chunk_size * i - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = halfn;

    // Read grmA
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        //read_grmA_oneCPU_forrt(file, startend(0, i), startend(1, i), var);
        read_grmA_oneCPU_forrt_withmiss(file, chunk_ranges(0, i), chunk_ranges(1, i), var);
    }

    // Compute grmAB chunk range
    int64_t lower_count = n * (n + 1) / 2 - upper_count;
    chunk_size = lower_count / chunks;
    chunk_ranges(0, 0) = halfn;
    for(int i = 1; i < chunks; i++) {
        chunk_ranges(0, i) = floor(sqrt(2 * (chunk_size * i + upper_count)  - 0.25) + 0.5);
    }
    chunk_ranges.row(1).segment(0, chunks - 1) = chunk_ranges.row(0).segment(1, chunks - 1);
    chunk_ranges(1, chunks - 1) = n;

    // Read grmAB
    #pragma omp parallel for num_threads(chunks)
    for(int i = 0; i < chunks; i++) {
        read_grmAB_oneCPU_forrt_withmiss(file, chunk_ranges(0, i), chunk_ranges(1, i), var);
    }
}

struct MappedFile {
  void *addr = nullptr;
  size_t size = 0;

  bool valid() { return addr != nullptr && size != 0; }

  void *memory_bound() {
      return (unsigned char*)addr + size;
  }

  void unmap() {
    munmap(addr, size);
    addr = nullptr;
    size = 0;
  }
};

MappedFile mmap_file(const char *filename) {
  MappedFile mapped_file;
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    spdlog::error("open file failed, path={}", filename);
    return mapped_file;
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    spdlog::error("failed to get file size, path={}", filename);
    return mapped_file;
  }
  size_t filesize = st.st_size;

  if (filesize % sizeof(float) != 0) {
    spdlog::error("file size is not a multiple of sizeof(float)!");
    return mapped_file;
  }

  void *mapped_addr = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);

  close(fd);

  if (mapped_addr == MAP_FAILED) {
    spdlog::error("failed to mmap file, path=", filename);
    return mapped_file;
  }

  mapped_file.addr = mapped_addr;
  mapped_file.size = filesize;

  return mapped_file;
}

void read_grmA_oneCPU_forrt_withmiss_v2(MappedFile mapped, long long start, long long end, float var){
    long long loclast;
    float f_buf = 0.0;
    for(int i = start; i < end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        long long nomiss_i = (long long)(nomissgrmid[i]);
        loclast = nomiss_i * (nomiss_i + 1) / 2;
        float* values = reinterpret_cast<float*>(mapped.addr) + loclast;
        for(int j = 0; j <= i; j++){
            f_buf = values[nomissgrmid[j]];
            if(i == j) _diag(i) += f_buf * var;
            else _A(i,j) += f_buf * var;
        }
    }
}

void read_grmAB_oneCPU_forrt_withmiss_v2(MappedFile mapped, long long start, long long end, float var){
    int halfn = (n + 1)/2;
    long long loclast;
    float f_buf = 0.0;
    for(int i = start; i < end; i++){
        if(i % 10000 == 0) spdlog::info("Reading id: {}", i);
        long long nomiss_i = (long long)(nomissgrmid[i]);
        loclast = nomiss_i * (nomiss_i + 1) / 2;
        float* values = reinterpret_cast<float*>(mapped.addr) + loclast;
        for(int j = 0; j <= i; j++){
            f_buf = values[nomissgrmid[j]];
            if(i == j) _diag(i) += f_buf * var;
            else if(j < halfn) _B(i - halfn,j) += f_buf * var;
            else _A(j - halfn, i - halfn) += f_buf * var;
        }
    }
}

/// read_grmAB_forrt_parallel with mmap on Linux
void read_grmAB_forrt_parallel_v2(string file, float var){
    PerfTimer _perf_timer("read grmAB forrt parallel v2");
    const long long halfn = (n + 1)/2;
    const int core_num = omp_get_num_procs();
    long long upper_count = halfn * (halfn + 1) / 2;
    long long chunk_size = upper_count / core_num;
    MatrixXi chunk_range(2, core_num);
    chunk_range(0, 0) = 0;

    for(int i = 1; i < core_num; i++){
        // x(x+1)/2 ~= i*chunk_size
        chunk_range(0, i) = floor(sqrt(2 * chunk_size * i - 0.25) + 0.5);
    }
    chunk_range.row(1).segment(0, core_num - 1) = chunk_range.row(0).segment(1, core_num - 1);
    chunk_range(1, core_num - 1) = halfn;

    MappedFile mapped = mmap_file(file.c_str());
    if (!mapped.valid()) return;

    #pragma omp parallel for
    for(int i = 0; i < core_num; i++){
        //read_grmA_oneCPU_forrt(file, startend(0, i), startend(1, i), var);
        read_grmA_oneCPU_forrt_withmiss_v2(mapped, chunk_range(0, i), chunk_range(1, i), var);
    }

    long long lower_count = n * (n + 1) / 2 - upper_count;
    chunk_size = lower_count / core_num;
    chunk_range(0, 0) = halfn;
    for(int i = 1; i < core_num; i++){
        chunk_range(0, i) = floor(sqrt(2 * (chunk_size * i + upper_count)  - 0.25) + 0.5);
    }
    chunk_range.row(1).segment(0, core_num - 1) = chunk_range.row(0).segment(1, core_num - 1);
    chunk_range(1, core_num - 1) = n;

    #pragma omp parallel for
    for(int i = 0; i < core_num; i++){
        //read_grmAB_oneCPU_forrt(file, startend(0, i), startend(1, i), var);
        read_grmAB_oneCPU_forrt_withmiss_v2(mapped, chunk_range(0, i), chunk_range(1, i), var);
    }

    mapped.unmap();
}


//20240409 20240430change 20240520 20250530
void large_randtr(const std::string& mhefile, const std::string& grmlist, const MatrixXf& ymat, int numofrt, int yid) {
    spdlog::info("The number of random vector is: {}", numofrt);
    vector<string> grms;
    string grmitem;
    std::ifstream fin_grm_list(grmlist, ios::in);
    if (!fin_grm_list.is_open()) spdlog::error("can not open the file phe");
    while (std::getline(fin_grm_list, grmitem)) {
        grms.push_back(grmitem);
        spdlog::info("{}", grmitem);
    }
    fin_grm_list.close();
    r = grms.size();

    spdlog::info("These {} GRMs (except the error) are included in the model.", r);

    VectorXf y = ymat.col(0);
    VectorXf varcmp(r + 1);
    VectorXf R1(r + 1), R2(r + 1);
    VectorXf Viy(n), ViViy(n);
    MatrixXf Vix(n, numofrt), Ax(n, numofrt), AI(r + 1, r + 1), AViy(n, r + 1), ViAViy(n, r + 1);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), AId, AIi;
    MatrixXf xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });

    std::ifstream fin_mhe(mhefile, ios::in);
    if (!fin_mhe.is_open()) spdlog::error("can not open the file phe");
    std::string line, s;
    int line_count = 0;
    while (std::getline(fin_mhe, line)) {
        line_count++;
        if (line_count == yid) {
            istringstream is(line);
            for (int i = 0; i <= r; i++){
                is >> s;
                varcmp(i) = stof(s);
            }
            break;
        }
    }
    fin_mhe.close();
    spdlog::info("The initial values are: {}\n", varcmp.transpose());

    int halfn = (n + 1) / 2;
    _singlebin.resize(1);
    _singlebin[0].A.setZero(halfn, halfn);
    _singlebin[0].B.setZero(halfn, halfn);

    int loopnum = 1; // TODO(wangkai) test 1 loop
    MatrixXf varcmpmat(loopnum, r + 1);
    _check = false;
    const int k_max_threads = omp_get_max_threads();
    spdlog::info("max openmp threads number is {}", k_max_threads);
    for (int loop = 0; loop < loopnum; loop++){
        PerfTimer perf_timer("calculating iteration " + std::to_string(loop + 1));

        _A.setZero(halfn, halfn);
        _B.setZero(halfn, halfn);
        _diag.setZero(n);

        for (int i = 0; i < r; i++) {
            spdlog::info("Reading the {} GRM for calculating V of iteration {}", getOrdinal(i + 1), loop + 1);
            // read_grmAB_forrt_parallel_v2(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
            read_grmAB_forrt_parallel(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
        }

        perf_timer.elapsed("read grmAB forrt");

        spdlog::info("calculating Vix of the random vectors");

        #pragma omp parallel for
        for (int i = 0; i < numofrt; i++) {
            Vix.col(i) = conjugate(1.0, varcmp(r), xs.col(i), 1);
        }
        Viy = conjugate(1.0, varcmp(r), y, 1);

        perf_timer.elapsed("Viy computation");

        _singlebin.resize(1);
        std::swap(_singlebin[0].A, _A);
        std::swap(_singlebin[0].B, _B);
        std::swap(_singlebin[0].diag, _diag);  // Backup

        perf_timer.elapsed("backup A and B");

        spdlog::info("Reading GRMs for calculating Aix of iteration {}", loop + 1);
        for (int i = 0; i < r; i++) {  //read second time
            read_grmAB_faster_parallel(grms[i] + ".grm.bin");
            spdlog::info("calculating A{}x of the random vectors", i + 1);

            #pragma omp parallel for
            for (int j = 0; j < numofrt; j++) {
                Ax.col(j) = Actimesx(xs.col(j), 1);
            }

            R1(i) = Ax.cwiseProduct(Vix).sum();
            AViy.col(i) = Actimesx(Viy, 1);
            R2(i) = Viy.dot(AViy.col(i));
        }

        perf_timer.elapsed("AViy computation");

        std::swap(_A, _singlebin[0].A);
        std::swap(_B, _singlebin[0].B);
        std::swap(_diag, _singlebin[0].diag); // Restore
        AViy.col(r) = Viy;

        perf_timer.elapsed("Restore A and B");

        #pragma omp parallel for
        for (int i = 0; i <= r; i++) {
            ViAViy.col(i) = conjugate(1.0, varcmp(r), AViy.col(i), 1);
        }

        perf_timer.elapsed("ViAViy.col(i) computation");

        R1(r) = xs.cwiseProduct(Vix).sum();
        R2(r) = Viy.dot(Viy) + C / varcmp(r);

        for (int i = 0; i < r + 1; i++) {
            for (int j = 0; j <= i; j++) {
                float value = AViy.col(i).cwiseProduct(ViAViy.col(j)).sum();
                AI(i, j) = value;
                AI(j, i) = value;
            }
        }

        R1 /= numofrt;
        AId = AI.cast<double>();
        AIi = AId.ldlt().solve(Imatd);
        varcmp -= AIi.cast<float>() * (R1 - R2);
        spdlog::info("The variance estimates after iteration {}  are: {}", loop + 1, varcmp.transpose());
        varcmpmat.row(loop) = varcmp.transpose();
        perf_timer.stop();
        std::cout << "\n\n";
    }
    cout  << endl << "The variance estimates of each iteration are: " << endl ;
    int col_width = 10;
    for (int i = 0; i < r; ++i) {
        std::cout << std::setw(col_width) << "V(G" + std::to_string(i+1) + ")";
    }
    std::cout << std::setw(col_width) << "V(E)" << std::endl << varcmpmat << endl;
    cout  << endl << "The heritability estimates are:" << endl;
    varcmp = varcmp / varcmp.sum();
    col_width = 13;
    cout << std::left;
    cout << setw(7) << "Group"
         << setw(col_width) << "Heritability"
         << setw(col_width) << "SE" << endl;

    for (int i = 0; i < r; ++i) {
        cout << setw(7) << "G" + std::to_string(i+1)
             << setw(col_width) << varcmp(i)
             << setw(col_width) << sqrt(AIi(i, i)) << endl;
    }

    cout << setw(7) << "Sum"
         << setw(col_width) << 1.0 - varcmp(r)
         << setw(col_width) << sqrt(AIi(r, r)) << endl;

    cout << std::right;

    cout  << endl << "The variance-covariance matrix is: " << endl  << AIi.cast<float>() << endl << endl ; //250514
}

//20240525
void randtr_small(string mhefile, string grmlist, MatrixXf ymat, int numofrt, int yid){
    cout << "The number of random vector is: " << numofrt << endl;
    vector<string> grms;
    string grmitem;
    ifstream list(grmlist, ios::in);
    if (!list.is_open()) cout << "can not open the file phe\n";
    do {
        getline(list,grmitem);
        grms.push_back(grmitem);
        cout << grmitem << endl;
    } while (!list.eof());
    list.close();
    r = grms.size();
    cout << "There are "<<  r <<" groups (except the error)"  << endl;

    clock_t start, end;
    VectorXf y = ymat.col(0); //20240624
    VectorXf varcmp(r + 1);
    VectorXf R1(r + 1), R2(r + 1);
    VectorXf Viy(n), ViViy(n);
    MatrixXf Vix(n, numofrt), Ax(n, numofrt), AI(r + 1, r + 1), AViy(n, r + 1), ViAViy(n, r + 1);
    MatrixXd Imatd = MatrixXd::Identity(r + 1, r + 1), AId, AIi;
    MatrixXf xs = MatrixXf::Zero(n, numofrt).unaryExpr([](float val) { return Rademacher(val); });


    ifstream infile(mhefile, ios::in); //read initial value of varcmp
    if (!infile.is_open()) cout << "can not open the file phe\n";
    string line, s;
    int lineCount = 0;
    while (getline(infile, line)) {
        lineCount++;
        if (lineCount == yid) {
            istringstream is(line);
            for (int i = 0; i <= r; i++){
                is >> s;
                varcmp(i) = stof(s);
            }
            break;
        }
    }
    infile.close();
    cout << "The initial value by HE is: " << varcmp.transpose() << endl;
    


    int halfn = (n + 1)/2;

    int loopnum = 2;
    MatrixXf varcmpmat(loopnum, r + 1);
    for (int loop = 0; loop < loopnum; loop++){
        _A.setZero(halfn, halfn);
        _B.setZero(halfn, halfn);
        _diag.setZero(n);
        start = clock();

        for (int i = 0; i < r; i++) {
            cout << "Reading the " << i + 1 <<"th GRM for V" << endl;
            read_grmAB_forrt_parallel(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
            //read_grmAB_forrandtr(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
        }
        if(loop == 0) _check = true;
        //Viy = conjugate(1.0, varcmp(r), y, 1);
        cout << "calculating Vix of the random vector: " << endl;
#pragma omp parallel for
        for (int i = 0; i <= numofrt; i++) {
            cout << i + 1 << " " << endl;
            if(i < numofrt)
               Vix.col(i) = conjugate(1.0, varcmp(r), xs.col(i), 1);
            else
                Viy = conjugate(1.0, varcmp(r), y, 1);
        }
        //cout << Vix.topLeftCorner(5, 5) << endl;
        _check = false;

        for (int i = 0; i < r; i++) {  //read second time
            read_grmAB_faster_parallel(grms[i] + ".grm.bin");
            //read_grmAB_faster(grms[i] + ".grm.bin");
            cout << "calculating A" << i + 1 << "x of the random vector: " << endl;
#pragma omp parallel for
                for (int j = 0; j < numofrt; j++) {
                    cout << j + 1 << " ";
                    Ax.col(j) = Actimesx(xs.col(j), 1);
                }
            cout << endl;
            R1(i) = Ax.cwiseProduct(Vix).sum();
            AViy.col(i) = Actimesx(Viy, 1);
            R2(i) = Viy.dot(AViy.col(i));
        }

        _A.setZero(halfn, halfn);
        _B.setZero(halfn, halfn);
        _diag.setZero(n);
        start = clock();

        for (int i = 0; i < r; i++) {
            cout << "Reading the " << i + 1 <<"th GRM for V" << endl;
            read_grmAB_forrt_parallel(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
            //read_grmAB_forrandtr(grms[i] + ".grm.bin", varcmp(i)); //read first time to calculate V
        }

        AViy.col(r) = Viy;
#pragma omp parallel for
        for (int i = 0; i <= r; i++) {
            ViAViy.col(i) = conjugate(1.0, varcmp(r), AViy.col(i), 1);
        }
        R1(r) = xs.cwiseProduct(Vix).sum();
        R2(r) = Viy.dot(Viy) + C / varcmp(r);
        

        for (int i = 0; i < r + 1; i++) {
            for (int j = 0; j <= i; j++) {
                AI(i, j) = AI(j, i) = AViy.col(i).cwiseProduct(ViAViy.col(j)).sum();
            }
        }
        R1 /= numofrt;
        //cout << R1.transpose() << endl;
        //cout << R2.transpose() << endl;
        AId = AI.cast<double>();
        AIi = AId.ldlt().solve(Imatd);
        varcmp -= AIi.cast<float>() * (R1 - R2);
        //cout << R1 << endl;
        //cout << R2 << endl;
        //cout << AI << endl;
        cout << varcmp.transpose() << endl;
        varcmpmat.row(loop) = varcmp.transpose();
        if(loop == 0){
            end = clock();
            cout <<  "1 iteration using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
        }
    }
    cout  << "The REMLranftr is: " << endl  << varcmpmat << endl;
}


VectorXf read_phe_search_commonid(string phefile, int whichy){
    vector<int> yid_vec;    // whole phe id
    vector<float> yraw_vec; // whole phe value
    string line, token;
    vector<float> nomissy;
    string index,pheitem,s;
    int i,j,temp,indextemp;
    ifstream phe(phefile, ios::in);
    if (!phe.is_open()) spdlog::error("can not open the file phe");
    while (std::getline(phe, line)) {
        istringstream is(line);
        is >> s; is >> s;
        yid_vec.push_back(stoi(s)); // individual ids
        // Skip to whichy
        for (j = 0; j < whichy; j++) {
          is >> s;
        }
        yraw_vec.push_back(stof(s)); // phenotype value
    }
    int nraw = yid_vec.size();
    VectorXi yid = Eigen::Map<VectorXi>(yid_vec.data(), nraw);
    VectorXf yraw = Eigen::Map<VectorXf>(yraw_vec.data(), nraw);
    nomissy.clear(); nomissgrmid.clear();

    // phenotype value can be missed (-9), we should filter them out.
    for (i = 0; i < ngrm; i++) {
        indextemp = grmid[i];
        for (j = 0; j < nraw; j++) {
            if(yid(j) == indextemp){
                if (!compare_float(yraw(j), -9.0)) {
                    nomissy.push_back(yraw(j));
                    grmidwithy[i] = true;
                    nomissgrmid.push_back(i);
                }
                break;
            }
        }
    }
    phe.close();
    yraw.resize(0);

    // n is the number of nomiss data.
    n = nomissy.size();

    spdlog::info("The sample size with non-missing phenotype value is: {}", n);
    //cout << nomissgrmid.size() << endl;
    VectorXf yscale = Eigen::Map<VectorXf>(nomissy.data(), n);
    yscale -= VectorXf::Constant(n, yscale.mean());
    yscale /= std::sqrt(yscale.squaredNorm() / (n - 1));
    return yscale;
}

//20240623
MatrixXf read_phe_search_formhe(string phefile){
    int nraw = 50000;  //这个值在hpc中改为表现型的实际总个体数！！！
    bool match;
    int count = 0;
    VectorXi yid(nraw);
    string index,pheitem,s;
    int i,j,temp,indextemp;
    ifstream phe(phefile, ios::in);
    if (!phe.is_open()) cout << "can not open the file phe\n";
    getline(phe,pheitem);
    istringstream is(pheitem);
    while (is >> s) {
        count++;
    }
    cout << count << endl;
    MatrixXf yraw = MatrixXf::Zero(nraw, count - 2);
    MatrixXf ymattemp = yraw;
    phe.seekg(0, ios::beg);
    for (i = 0; i < nraw; i++) {
        getline(phe,pheitem);

        istringstream is(pheitem);
        is >> s; is >> s;
        yid(i) = stoi(s);
        for (j = 0; j < count - 2; j++) {
            is >> s;
            yraw(i, j) = stof(s);
        }
    }
    nomissgrmid.clear();
    count = 0;
    for (i = 0; i < ngrm; i++) {
        indextemp = grmid[i];
        match = false;
        for (j = 0; j < nraw; j++) {
            if(yid(j) == indextemp){
                ymattemp.row(count) = yraw.row(j);
                count++;
                grmidwithy[i] = true;
                nomissgrmid.push_back(i);
                match = true;
                break;
            }
        }
        if (!match)
        grmidwithy[i] = false;
    }
    phe.close();
    yraw.resize(0, 0);

    n = nomissgrmid.size();
    cout << "n is: " << n << endl;
    MatrixXf ymat = ymattemp.topRows(n);
    cout << ymat.topRows(5) << endl;
    VectorXf yytemp = VectorXf::Zero(n);
    for (j = 0; j < ymattemp.cols(); j++) {
        yytemp = ymat.col(j);
        yytemp -= VectorXf::Constant(n, yytemp.mean());
        yytemp /= (yytemp.norm() / sqrt(n - 1));
        ymat.col(j) = yytemp;
    }
    ymattemp.resize(0, 0);
    return ymat;
}

struct MpheReader {
    void operator()(const std::string &name, const std::string &value, Mphe &mphe) {
        std::vector<std::string> values = split_string(value, ',');
        assert(values.size() == 2);
        mphe.i = std::stoi(values[0]);
        mphe.path = values[1];
    }
};



int main(int argc, const char * argv[]) {
    // insert code here...

    PerfTimer _perf_timer("total");

    args::ArgumentParser arg_parser("fastgreml");
    args::HelpFlag help(arg_parser, "help", "Display help menu", {'h', "help"});
    args::CompletionFlag completion(arg_parser, {"complete"});
    args::ValueFlag<std::string> arg_grmlist_path(arg_parser, "GRM list path", "GRM list path", {'g', "grmlist"});
    args::ValueFlag<Mphe, MpheReader> arg_mphe(arg_parser, "MPHE", "MPHE", {'m', "mphe"});
    args::ValueFlag<std::string> arg_cov_path(arg_parser, "UKB covariates", "UKB covariates", {'c', "cov"});
    args::ValueFlag<std::string> arg_init_values(arg_parser, "Initial values", "Initial values", {'i', "initial"});
    args::ValueFlag<std::string> arg_output_path(arg_parser, "Output path", "Output path", {'o', "output"});

    try {
        arg_parser.ParseCLI(argc, argv);
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << arg_parser;
        return 0;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << arg_parser;
        return 1;
    }

    std::string grmlist_path = args::get(arg_grmlist_path);
    Mphe mphe = args::get(arg_mphe);
    std::string cov_path = args::get(arg_cov_path);
    std::string init_values = args::get(arg_init_values);
    std::string output_path = args::get(arg_output_path);

    std::cout << "\n\nAll args: \n" << "grmlist_path: " << grmlist_path << '\n'
              << "mphe_path: " << mphe.path << ",\n"
              << "           and the " << getOrdinal(mphe.i) << " phenotype will be analyzed." << '\n'
              << "cov_path: " << cov_path << '\n'
              << "init_value_path:" << init_values << '\n'
              << "output_path: " << output_path << "\n\n";

    int yid = mphe.i;

    VectorXf aaa,bbb,ccc;
    struct rusage usage;
    //   string grmfile = "/storage/yangjianLab/baiweiyang/SV_Imputation_Project_final/GREML_74/GRM_for_greml/LDref.EUR.maf0.01.hwe-6.chrAuto.SNV";
 //     string phefile = "/storage/yangjianLab/xuting/data/phe/47.pheno";
    //string phefile = "/storage/yangjianLab/chenshuhua/project/WES/UKB_pheno/PHESANT_pheno/dat1/Continuous/50.pheno";
     // string phefile = "/storage/yangjianLab/chenshuhua/project/WES/UKB_pheno/PHESANT_pheno/dat3/Continuous/23105.pheno";
   // string phefile = "/storage/yangjianLab/baiweiyang/SV_Imputation_Project_final/PHENOTYPE/Continuous_final_652/23105.pheno";
    string grmfile = "/home/kai/WestlakeProjects/ldms-data/group1";

    // string grmfile = "/storage/yangjianLab/xuting/data/grm/WGS_unrel/sample50k/grmhe1_nml_noIG";
    string grmlist = grmlist_path;
    string covfile = cov_path;
    string mphefile = mphe.path;
    string mhefile = init_values;

    clock_t start, end;
    start = clock();
    // MatrixXf carrot;
    // carrot.setZero(260000, 260000);
    // getrusage(RUSAGE_SELF, &usage);
    // cout << "Memory usage: " << usage.ru_maxrss / 1024.0  << " MB" << endl;
    // end = clock();
    // cout <<  "carrot is ready, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;


    start = clock();

    std::ifstream fin_grmlist(grmlist);
    std::getline(fin_grmlist, grmfile);
    read_grmid(grmfile);
    //read_realphe_all(phefile);

    //VectorXf y = merge_pheid_grmid();
    end = clock();
    //cout <<  "merge is done, using " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    //cout <<  n << " indviduals are in common" <<endl;

    VectorXf y = read_phe_search_commonid(mphefile, yid);
    MatrixXf ymat(n, 1);
    ymat.col(0) = y;

    //read_realcov_ofcommonid(covfile);
    //n = 46567;
    //read_realcov_search(covfile);

    //read_grmlist(phefile, grmlist, covfile);
    read_realcov_search_withmiss_v2(covfile); //20240620
    // read_realcov_search_withmiss(covfile); //20240620

//    y = proj_x(y);
//    y -= VectorXf::Constant(n, y.mean());
//    y /= (y.norm() / sqrt(n - 1));

    getrusage(RUSAGE_SELF, &usage);
    spdlog::info("Memory usage: {} MB", usage.ru_maxrss / 1024.0);


    int grmmethod = 1;  //0 for grm, 1 for grmAB, 2 for sparsegrm, 3 for gemline
    start = clock();
    //readgrm(grmmethod, grmfile);
    end = clock();
    //cout <<  "grm is ready, using: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

    ymatproj(ymat);   //20240114
    //randtr_small(mhefile, grmlist, ymat, 39, corenumber);

    //testremlemhe(mphefile, grmlist, covfile, ymat, C, 2);
    //large_mhe_v3(grmlist, mhefile, ymat, 50);
    //large_remle(grmlist, mhefile, ymat, 29);
    //MatrixXf ab = mremlrandtr_mtraits(ymat, C, 50, 1);
    //cout << "the remltandtr is:"<< endl << ab << endl;
    large_randtr(mhefile, grmlist, ymat, 50, yid);

    //ymat.col(0) = ymat.col(5);
    //VectorXf rr = mremlrandtr(ymat.col(0), C, 30, 1);

    //mhe.resize(r + 1);
    //mhe << 0.05, 0.05,0.05,0.05,0.05, 0.75;
   // writemultiseeds(ymat, 0.5, 50, 29, seed240120);
    //read_multiLanzcos(ymat, C, 0.5, 50, 29, multiseedfile);
   // testmulti_seedsforsingle(ymat, C, 0.5, 50, 29, multiseedfile);
    //VectorXf rrr = multiLanzcos(ymat, C, 0.5, 30, 1, seedfile);
//    VectorXf rr;
//    cout <<  "mreml: " << (rr / rr.sum()).transpose() << endl;
//    rr = mhe;
//    cout << "he: " << (rr / rr.sum()).transpose() << endl;
//    rr = remlstc;
//    cout << "remlstc: " << (rr / rr.sum()).transpose() << endl;
//    rr = remlestc;
//    cout << "remlestc: " << (rr / rr.sum()).transpose() << endl;
   // cout << "multilanczos: " << (rrr / rrr.sum()).transpose() << endl;
    //VectorXf test = testformulLanzcos(ymat, C, 0.5, 50, 6, seedfile);
    //cout << "the estimated heritability is: " << test.transpose() << endl;

    //vector<Lseed> seeds = readseeds(0.5, 50, 6,seedfile);
     //VectorXf h2s = remlreg_seeds(ymat, C, 0.5, 50, 6, seedfile);
   // cout << "the estimated heritability is: " << h2s << endl;

    getrusage(RUSAGE_SELF, &usage);
    spdlog::info("Memory usage: {} MB", usage.ru_maxrss / 1024.0);

    //cout << "sum(grm) = " << (_A.sum()+_B.sum())*2+_diag.sum() << endl;
    //cout << "tr(grm) = " << _diag.sum() << endl;

    //cout << _diag.head(100) << endl;

    Eigen::Vector2f r3;

    //r3 = remlesavememory(y, C, grmmethod);
    //cout << "the estimated heritability is: " <<r3(0)/r3.sum() << endl;

    //r3 = remlerandtrwithlanczos(y, C, 0.6, 10);
    //cout << "the estimated heritability is: " <<r3(0)/r3.sum() << endl;

    return 0;
}
