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
    cin >> n;
    for (int i = 0; i < n; ++i) {
        int k = 0;
        if (i < (n / 2 + 1)) {
            k = i;
        } else {
            k = n - 1 - i;
        }
        for (int j = 1; j <= n; ++j) {
            if (abs(j - (n + 1) / 2) <= k) {
                cout << 'D';
            } else {
                cout << '*';
            }
        }
        cout << endl;
    }
    return 0;
}
