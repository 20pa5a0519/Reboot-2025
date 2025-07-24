import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Results } from '../results';
import { RouterOutlet, RouterLink } from '@angular/router';

@Component({
  standalone: true,
  selector: 'app-input',
  imports: [RouterLink, RouterOutlet, FormsModule, CommonModule],
  templateUrl: './input.html',
  styleUrl: './input.css',
})
export class Input {
  constructor(private api: Results) {}

  transactions: any = [];
  length: number = 0;
  legitimate_count: any = null;
  mule_count: any = null;
  kyc_invalid_count = 0;
  riskScore: number | null = null;
  notFound: number | null = null;

  errorMessage = '';

  accountId = '';
  phone = '';
  result: number | null = null;

  detect() {
    this.result = null;
    this.notFound = null;
    //sending Id to backend
    this.api.searchAccount(this.accountId).subscribe(
      (response) => {
        console.log(this.accountId, ' sent');
        console.log('ID sent');

        //fetchig results from backend
        this.api.fetchResults().subscribe(
          (results) => {
            if (results && results.length > 0) {
              console.log('fetched results');
              //store in service
              this.api.transactions = results;
              //Persist in localStorage
              localStorage.setItem('transactions', JSON.stringify(results));
              //store locally
              const storedTransactions = localStorage.getItem('transactions');
              if (storedTransactions) {
                this.transactions = JSON.parse(storedTransactions);
              }
              this.processTransactions();
              this.errorMessage = '';
              this.notFound = 0;
              console.log(this.transactions);
            } else {
              this.transactions = [];
              this.api.transactions = [];
              this.notFound = 1;
              this.errorMessage = 'Account Not Found in database';
            }
            console.log('notFound value updated -', this.notFound);
          },
          (err) => {
            this.transactions = null;
            this.errorMessage = err.error?.message || 'error fetching result';
            console.log('error fetching results', err);
          }
        );
      },
      (error) => {
        console.log('error sending ID');
        console.log(error);
      }
    );
  }

  //processing transactions
  processTransactions() {
    this.length = this.transactions.length;
    this.legitimate_count = 0;
    this.mule_count = 0;
    this.kyc_invalid_count = 0;
    this.riskScore = null;
    this.result = null;

    console.log('Starting evaluation, transactions:', this.transactions);

    if (this.length > 1) {
      let riskSum = 0;

      for (let txn of this.transactions) {
        console.log(
          'multiple transactions,Checking txn.fraud_flag =',
          txn.Predicted_Fraud_Flag
        );
        if (txn.Predicted_Fraud_Flag === 1) {
          this.mule_count!++;
          //kyc counting
          if (txn.KYC_Fingerprinting === 'Invalid') {
            this.kyc_invalid_count++;
          }
          //sum risk score
          riskSum += Number(txn.Predicted_Risk_Score) || 0;
          console.log('risk sum', riskSum);
        } else {
          this.legitimate_count!++;
        }
      }
      // If any fraudulent records exist, treat overall result as mule
      this.result = this.mule_count! > 0 ? 1 : 0;
      this.riskScore = Math.round(riskSum / this.mule_count);
      console.log(
        `Multiple txns processed — mule_count=${this.mule_count}, legitimate_count=${this.legitimate_count}, result=${this.result}`
      );
      console.log('risk score avg', this.riskScore);
    } else if (this.length === 1) {
      const txn = this.transactions[0];
      console.log('Single txn.Fraud_Flag =', txn.Predicted_Fraud_Flag);
      if (txn.Predicted_Fraud_Flag === 1) {
        this.result = 1;
        this.mule_count = 1;
        this.legitimate_count = 0;
        if (txn.KYC_Fingerprinting === 'Invalid') {
          this.kyc_invalid_count++;
        }
        this.riskScore = txn.Predicted_Risk_Score;
      } else {
        this.result = 0;
        this.mule_count = 0;
        this.legitimate_count = 1;
      }

      console.log(
        `Single txn processed — result=${this.result}, mule_count=${this.mule_count}, legitimate_count=${this.legitimate_count}`
      );
    } else {
      // No transactions found—handle as needed
      this.result = null;
      console.log('No transactions found');
    }
    console.log('Final result:', this.result);
  }
}
