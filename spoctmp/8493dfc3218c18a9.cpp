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
    int x1, x2, y1, y2;
    int diff1, diff2;
    std::cin >> x1 >> y1;
    std::cin >> x2 >> y2;
    diff1 = std::abs(x2 - x1);
    diff2 = std::abs(y2 - y1);
    if (diff1 > diff2) {
        std::cout << diff1;
    } else if (diff2 > diff1) {
        std::cout << diff2;
    } else {
        std::cout << diff1;
    }
    return 0;
}
