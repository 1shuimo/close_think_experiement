#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;

int main() {
    int k;
    int x[4];
    std::cin >> x[1] >> x[1] >> x[2];
    std::sort(x, x + 3);
    k = x[2] - x[0];
    std::cout << k;
    return 0;
}
