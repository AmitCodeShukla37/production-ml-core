#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int>req(numCourses,0);
        vector<vector<int>>curriculum(numCourses);

        for(auto x : prerequisites){
            int u = x[0], v = x[1];
            curriculum[v].push_back(u);
            req[u]++;
        }

        int completed = 0;
        queue<int>q;

        for(int i = 0; i < numCourses; i++){
            if(!req[i]){
                q.push(i);
            }
        }

        while(!q.empty()){
            int u = q.front(); q.pop();
            completed++;
            for(auto v : curriculum[u]){
                req[v]--;
                if(!req[v]){
                    q.push(v);
                }
            }
        }

        return completed == numCourses;
    }
};