import { Routes } from '@angular/router';
import { Home } from './home/home';
import { Login } from './login/login';

import { Reports } from './reports/reports';
import { Input } from './input/input';
import { Details } from './details/details';
import { Details2 } from './details2/details2';
import { Graph } from './graph/graph';
import { Live } from './live/live';

export const routes: Routes = [
  {
    path: '',
    component: Home,
  },

  {
    path: 'login',
    component: Login,
  },
  {
    path: 'home',
    component: Home,
  },
  {
    path: 'input',
    component: Input,
  },
  {
    path: 'reports',
    component: Reports,
  },
  {
    path: 'details/:id',
    component: Details,
  },
  {
    path: 'details2/:id',
    component: Details2,
  },
  {
    path: 'graph/:id',
    component: Graph,
  },
  {
    path: 'live',
    component: Live,
  },
];
