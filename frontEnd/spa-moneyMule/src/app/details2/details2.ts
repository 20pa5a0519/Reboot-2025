import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Results } from '../results';

@Component({
  selector: 'app-details2',
  imports: [FormsModule, CommonModule],
  templateUrl: './details2.html',
  styleUrl: './details2.css',
})
export class Details2 {
  accountId: String = '';

  constructor(private route: ActivatedRoute, private api: Results) {}

  transactions: any = [];
  length: number = 0;
  legitimate_count: any = null;
  mule_count: any = null;
  kyc_invalid_count = 0;
  riskScore: number | null = null;
  notFound: number | null = null;
  filteredTransactions: any = [];
  reasons : any = [];

  errorMessage = '';

  ngOnInit() {
    this.accountId = this.route.snapshot.paramMap.get('id')!;


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

              this.api.searchReasons(this.accountId).subscribe(
                () => {
                  this.api.getReasons().subscribe((res: any) => {
                    this.reasons = res;
                    console.log('Reasons fetched:', this.reasons);
                  });
                },
                (err) => {
                  console.error('Error in searchReasons:', err);
                }
              );

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
  processTransactions() {
    this.length = this.transactions.length;
    this.legitimate_count = 0;
    this.mule_count = 0;
    this.kyc_invalid_count = 0;
    this.riskScore = null;

    this.filteredTransactions = this.transactions.filter(
      (t: any) => t.Predicted_Fraud_Flag === 1
    );

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

      this.riskScore = Math.round(riskSum / this.mule_count);
      console.log(
        `Multiple txns processed — mule_count=${this.mule_count}, legitimate_count=${this.legitimate_count}`
      );
      console.log('risk score avg', this.riskScore);
    } else if (this.length === 1) {
      const txn = this.transactions[0];
      console.log('Single txn.Fraud_Flag =', txn.Predicted_Fraud_Flag);
      if (txn.Predicted_Fraud_Flag === 1) {
        this.mule_count = 1;
        this.legitimate_count = 0;
        if (txn.KYC_Fingerprinting === 'Invalid') {
          this.kyc_invalid_count++;
        }
        this.riskScore = txn.Predicted_Risk_Score;
      } else {
        this.mule_count = 0;
        this.legitimate_count = 1;
      }

      console.log(
        `Single txn processed — mule_count=${this.mule_count}, legitimate_count=${this.legitimate_count}`
      );
    } else {
      // No transactions found—handle as needed
      console.log('No transactions found');
    }

    this.loadTransactions();
  }

  // Pagination state
  pageSize = 25;
  currentPage = 1;
  totalItems = 0;
  pagedTransactions: any[] = [];

  get totalPages(): number {
    return Math.ceil(this.totalItems / this.pageSize) || 1;
  }

  get startIndex(): number {
    return (this.currentPage - 1) * this.pageSize;
  }

  get endIndex(): number {
    return Math.min(this.totalItems, this.currentPage * this.pageSize);
  }

  get pagesToShow(): number[] {
    const pages = [];
    for (let i = 1; i <= this.totalPages; i++) {
      if (
        i <= 3 ||
        i > this.totalPages - 2 ||
        Math.abs(i - this.currentPage) <= 1
      ) {
        pages.push(i);
      }
    }
    return pages;
  }

  loadTransactions() {
    // Suppose filteredTransactions is already populated
    this.totalItems = this.filteredTransactions.length;
    this.onPageChange(1);
  }

  onPageChange(page: number) {
    if (page < 1 || page > this.totalPages) return;
    this.currentPage = page;
    const start = this.startIndex;
    this.pagedTransactions = this.filteredTransactions.slice(
      start,
      start + this.pageSize
    );
  }
}
