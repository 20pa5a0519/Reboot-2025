import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink } from '@angular/router';
import { Results } from '../results';

interface Transaction {
  Sender_Account: string;
  Fraud_flag: 0 | 1;
  // other fields...
}

@Component({
  standalone: true,
  selector: 'app-reports',
  imports: [FormsModule, CommonModule, RouterLink, RouterOutlet],
  templateUrl: './reports.html',
  styleUrl: './reports.css',
})
export class Reports {
  constructor(private api: Results) {}

  allTransactions: any = [];

  uniqueAccounts: string[] = [];
  muleAccounts: string[] = [];
  legitAccounts: string[] = [];
  totalUnique = 0;
  muleCount = 0;
  legitCount = 0;

  ngOnInit() {
    const cachedData = localStorage.getItem('cachedReportsData');

    if (cachedData) {
      const parsed = JSON.parse(cachedData);
      this.muleAccounts = parsed.muleAccounts;
      this.legitAccounts = parsed.legitAccounts;
      this.muleCount = parsed.muleCount;
      this.legitCount = parsed.legitCount;
      this.totalUnique = this.muleCount + this.legitCount;
      console.log('Loaded reports data from localStorage');
    } else {
      this.api.fetchTransactions().subscribe(
        (data: any) => {
          if (data) {
            console.log('Fetched fresh data from backend');
            this.muleAccounts = data.muleAccounts;
            this.legitAccounts = data.legitAccounts;
            this.muleCount = data.muleCount;
            this.legitCount = data.legitCount;
            this.totalUnique = this.muleCount + this.legitCount;

            // Store to localStorage for future use
            const cachePayload = {
              muleAccounts: this.muleAccounts,
              legitAccounts: this.legitAccounts,
              muleCount: this.muleCount,
              legitCount: this.legitCount,
            };
            localStorage.setItem(
              'cachedReportsData',
              JSON.stringify(cachePayload)
            );
          }
        },
        (err) => {
          console.log('Error fetching transactions', err);
        }
      );
    }
  }

  // ngOnInit() {
  //   this.api.fetchTransactions().subscribe(
  //     (data: any) => {
  //       if (data) {
  //         console.log('fetched all results in reports');
  //         //store in service
  //         this.muleAccounts = data.muleAccounts; // Keep full account objects
  //         this.legitAccounts = data.legitAccounts;
  //         this.muleCount = data.muleCount;
  //         this.legitCount = data.legitCount;
  //         this.totalUnique = this.muleCount + this.legitCount;

  //         console.log('mule count', this.muleCount);
  //         //Persist in localStorage
  //         // localStorage.setItem('allTransactions', JSON.stringify(response));
  //         //store locally
  //         // const storedTransactions = localStorage.getItem('allTransactions');
  //         // if (storedTransactions) {
  //         //   this.allTransactions = JSON.parse(storedTransactions);
  //         // }
  //       }
  //     },
  //     (err) => {
  //       this.allTransactions = null;
  //       console.log('error fetching transactions', err);
  //     }
  //   );
  // }

  selectedView: 'mule' | 'legit' | null = null;
  tableData: { id: string; HolderName: string; status: string }[] = [];

  viewAccounts(type: 'mule' | 'legit') {
    this.selectedView = type;
    const accountList =
      type === 'mule' ? this.muleAccounts : this.legitAccounts;

    // Filter transactions associated with each account for display
    this.allData = accountList.map((acc: any) => ({
      id: acc.id,
      HolderName: acc.HolderName || 'N/A',
      status: type === 'mule' ? 'Mule' : 'Legit',
    }));

    this.totalItems = this.allData.length;
    this.currentPage = 1; // reset to first page
    this.refreshTable();
  }

  get legitPercent(): number {
    return this.totalUnique
      ? Math.round((this.legitCount / this.totalUnique) * 100)
      : 0;
  }

  get mulePercent(): number {
    return this.totalUnique
      ? Math.round((this.muleCount / this.totalUnique) * 100)
      : 0;
  }

  // processTransactions() {
  //   this.uniqueAccounts = [];
  //   this.muleAccounts = [];
  //   this.legitAccounts = [];
  //   this.muleCount = 0;
  //   this.legitCount = 0;

  //   const MAX_STORE = 100; // optional: limit accounts stored in localStorage

  //   const seenAccounts = new Set();

  //   for (const txn of this.allTransactions) {
  //     const acct = txn.Sender_Account;
  //     if (!seenAccounts.has(acct)) {
  //       seenAccounts.add(acct);

  //       const relatedTxns = this.allTransactions.filter(
  //         (t: any) => t.Sender_Account === acct
  //       );

  //       const isMule = relatedTxns.some(
  //         (t: any) => t.Predicted_Fraud_Flag === 1
  //       );
  //       if (isMule) {
  //         this.muleAccounts.push(acct);
  //       } else {
  //         this.legitAccounts.push(acct);
  //       }
  //     }
  //   }

  //   this.muleCount = this.muleAccounts.length;
  //   this.legitCount = this.legitAccounts.length;
  //   this.totalUnique = this.muleCount + this.legitCount;
  //   console.log('total : ', this.totalUnique);
  //   console.log('legit : ', this.legitCount);
  //   console.log('mule : ', this.muleCount);

  //   // ✅ Save only count and top N accounts (optional)
  //   localStorage.setItem('muleCount', this.muleCount.toString());
  //   localStorage.setItem('legitCount', this.legitCount.toString());

  //   localStorage.setItem(
  //     'muleAccountsSample',
  //     JSON.stringify(this.muleAccounts.slice(0, MAX_STORE))
  //   );
  //   localStorage.setItem(
  //     'legitAccountsSample',
  //     JSON.stringify(this.legitAccounts.slice(0, MAX_STORE))
  //   );
  // }

  // processTransactions() {
  //   this.uniqueAccounts = [];
  //   this.muleAccounts = [];
  //   this.legitAccounts = [];
  //   this.muleCount = 0;
  //   this.legitCount = 0;

  //   // 1. Build a list of unique Sender_Account IDs
  //   for (const txn of this.allTransactions) {
  //     const acct = txn.Sender_Account;
  //     if (!this.uniqueAccounts.includes(acct)) {
  //       this.uniqueAccounts.push(acct);
  //     }
  //   }

  //   this.totalUnique = this.uniqueAccounts.length;

  //   // 2–4. Iterate over each unique account ID
  //   for (const acct of this.uniqueAccounts) {
  //     // Collect all transactions for this account
  //     const relatedTxns = this.allTransactions.filter(
  //       (t: any) => t.Sender_Account === acct
  //     );

  //     if (relatedTxns.length === 1) {
  //       // Single transaction case
  //       const flag = relatedTxns[0].Predicted_Fraud_Flag;
  //       if (flag === 1) {
  //         this.muleCount++;
  //         this.muleAccounts.push(acct);
  //       } else {
  //         this.legitCount++;
  //         this.legitAccounts.push(acct);
  //       }
  //     } else {
  //       // Multiple transactions: check if any one is flagged
  //       let isMule = false;
  //       for (const t of relatedTxns) {
  //         if (t.Predicted_Fraud_Flag === 1) {
  //           isMule = true;
  //           break;
  //         }
  //       }
  //       if (isMule) {
  //         this.muleCount++;
  //         this.muleAccounts.push(acct);
  //       } else {
  //         this.legitCount++;
  //         this.legitAccounts.push(acct);
  //       }
  //     }
  //   }
  // }

  //pagination
  totalItems = 0;
  pageSize = 25;
  currentPage = 1;
  allData: any[] = [];

  get totalPages(): number {
    return Math.ceil(this.totalItems / this.pageSize);
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
        i <= 5 ||
        i > this.totalPages - 2 ||
        Math.abs(i - this.currentPage) < 2
      ) {
        pages.push(i);
      }
    }
    return pages;
  }

  onPageChange(page: number) {
    if (page < 1 || page > this.totalPages) return;
    this.currentPage = page;
    this.refreshTable();
  }

  refreshTable() {
    const start = this.startIndex;
    const end = this.endIndex;
    this.tableData = this.allData.slice(start, end);
  }

  get showViewDetailsColumn(): boolean {
    return this.tableData.some((acct) => acct.status === 'Mule');
  }
}
