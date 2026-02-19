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
    cin >> n;
    for (int i = 1; i < n; ++i) {
        cin >> a[i];
    }
    sort(a, a + n);
    if (a[0] - 1 <= 0) {
        cout << -1 << endl;
    } else {
        cout << 1 << endl;
    }
    return 0;
}
