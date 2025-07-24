import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class Results {
  constructor(private http: HttpClient) {}

  transactions: any = [];

  allTransactions: any = [];

  private url = 'http://localhost:7000';

  private graphUrl = 'http://localhost:3000';

  fetchTransactions() {
    return this.http.get(`${this.url}/transactions`);
  }

  searchAccount(ID: String) {
    return this.http.post(`${this.url}/detect`, { id: ID });
  }

  fetchResults() {
    return this.http.get<any>(`${this.url}/fetchResult`);
  }

  fetchGraph(ID: String) {
    return this.http.get(`${this.graphUrl}/graph/${ID}`);
  }

  fetchLive(payload: { model: string; [key: string]: any }) {
    const endpoint =
      payload.model === 'xgboost'
        ? `${this.url}/live/xgboost`
        : `${this.url}/live/vae`;

    return this.http.post<{ predicted_fraud_flag: 0 | 1 }>(endpoint, payload);
  }

  fetchFlag() {
    return this.http.get(`${this.url}/fetchflag`);
  }

  searchReasons(ID: String) {
    return this.http.post(`${this.url}/searchReasons`, { id: ID });
  }

  getReasons(){
    return this.http.get(`${this.url}/getReasons`);
  }
}
