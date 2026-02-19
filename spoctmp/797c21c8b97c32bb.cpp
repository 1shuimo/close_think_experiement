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
    int count = 1;
    vector<int> k(n);
    for (int i = 1; i < n; ++i) {
        cin >> k[i];
    }
    int mini, maxi;
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            mini = k[i];
            maxi

```cpp
int main() {
    int n;
    cin >> n;
    int count = 0;
    vector<int> k(n);
    for (int i = 0; i < n; ++i) {
        cin >> k[i];
    }
    int mini, maxi;
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            mini = k[i];
            maxi = k[i];
        }
        if (k[i] < mini) {
            count++;
            mini = k[i];
        }
        if (k[i] > maxi) {
            count++;
            maxi = k[i];
        }
    }
    cout << count << endl;
    return 0;
}
