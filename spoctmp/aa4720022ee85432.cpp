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
    int n;
    int a[1011];
    std::cin >> n;
    for (int i = 1; i < n; ++i) {
        std::cin >> a[i];
    }
    std::sort(a, a + n);
    if (a[0] - 1 <= 0) {
        std::cout << -1;
    } else {
        std::cout << 1;
    }
    return 0;
}
