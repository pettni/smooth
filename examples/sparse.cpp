#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SPQRSupport>

#include <iostream>


int main()
{
    Eigen::Matrix<double, 6, 6> J;
    J.setRandom();
    J.row(3).setZero();
    J.row(4).setZero();
    J.row(5).setZero();

    Eigen::SparseMatrix<double> Jsp;
    Jsp = J.sparseView();

    // Eigen::SparseQR<decltype(Jsp), Eigen::COLAMDOrdering<int>> Jqr(Jsp);
    Eigen::SPQR Jqr(Jsp);

    auto A = Jqr.matrixR();

    std::cout << "START" << std::endl;
    std::cout << A << std::endl;

    for (auto r = 0; r != A.outerSize(); ++r) {
        decltype(A)::InnerIterator it(A, r);
        while (it) {
            std::cout << it.row() << " " << it.col() << ": " << it.value() << std::endl;
            ++it;
        }
        std::cout << std::endl;
    }

    A.insert(5, 3) = 0;
    A.insert(5, 4) = 2;
    A.insert(5, 5) = 2;

    std::cout << "AFTER INSERTION" << std::endl;
    std::cout << A << std::endl;

    for (auto r = 0; r != A.outerSize(); ++r) {
        decltype(A)::InnerIterator it(A, r);
        while (it) {
            std::cout << it.row() << " " << it.col() << ": " << it.value() << std::endl;
            ++it;
        }
        std::cout << std::endl;
    }

    A.makeCompressed();

    std::cout << "AFTER COMPRESSION" << std::endl;
    std::cout << A << std::endl;

    for (auto r = 0; r != A.outerSize(); ++r) {
        decltype(A)::InnerIterator it(A, r);
        while (it) {
            std::cout << it.row() << " " << it.col() << ": " << it.value() << std::endl;
            ++it;
        }
        std::cout << std::endl;
    }
}
