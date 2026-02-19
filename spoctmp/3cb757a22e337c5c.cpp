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
    long long x1, y1, x2, y2, x, y;
    cin >> x1 >> y1 >> x2 >> y2 >> x >> y;
    if (llabs(x2 - x1) % x != 0 || llabs(y2 - y1) % y != 0) {
        cout << "NO\n";
        return 0;
    }
    if (((x2 - x1) / x - (y2 - y1) / y) % 2 == 0) {
        cout << "YES\n";
    } else {
        cout << "NO\n";
    }
    return 0;
}
