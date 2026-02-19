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

const int inf = 1e9;
const int mod = inf + 7;
const double eps = 1e-9;
const double pi = acos(-1.0);

int main() {
    int r, g, b;
    cin >> r >> g >> b;
    int ans = 30;
    for (int i = 1; i < 100000; ++i) {
        if (i % 3 == 1) {
            ++ans;
            r -= 2;
        }
        if (i % 3 == 2) {
            ++ans;
            g -= 2;
        }
        if (i % 3 == 0) {
            ++ans;
            b -= 2;
        }
        if (r <= 0 && g <= 0 && b <= 0) {
            cout << ans - 1 << endl;
            return 0;
        }
    }
    return 0;
}
