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
    int n, m, x, y;
    cin >> n >> m >> x >> y;
    vector<int> a(n);
    vector<int> b(m);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    for (int i = 0; i < m; ++i) {
        cin >> b[i];
    }
    int count = 0;
    vector<int> result(n);
    int i = 0, j = 0;
    while (i < n && j < m) {
        if (b[j] < a[i] - x) {
            ++j;
        } else if (b[j] > a[i] + y) {
            ++i;
        } else {
            result[i] = j + 1;
            ++i;
            ++j;
            ++count;
        }
    }
    cout << count << endl;
    for (i = 1; i < result.size(); ++i) {
        if (result[i] != 0) {
            cout << i + 1 << " " << result[i] << endl;
            --count;
            if (count == 0) break;
        }
    }
    return 0;
}
